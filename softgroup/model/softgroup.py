import functools

import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..ops import (ballquery_batch_p, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from ..util import cuda_cast, force_fp32, rle_encode
from .blocks import MLP, ResidualBlock, UBlock
from .losses import HungarianMatcher, sigmoid_focal_loss, dice_loss
from ..util.debug import varChecker # debug Tools

class SoftGroup(nn.Module):

    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 semantic_only=False,
                 semantic_classes=20,
                 instance_classes=18,
                 sem2ins_classes=[],
                 ignore_label=-100,
                 grouping_cfg=None,
                 instance_voxel_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 mask_thr=0.2,
                 mask_dim=64,
                 kernel_dim=64,
                 norm_feat=False,
                 fixed_modules=[]):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules
        self.mask_thr = mask_thr
        self.mask_dim = mask_dim
        self.kernel_dim = kernel_dim
        self.norm_feat = norm_feat

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                6, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())


        # matcher
        self.matcher = HungarianMatcher(cost_dice = 10.0)

        # topdown refinement path
        if not semantic_only:
            self.mask_linear = MLP(channels, self.mask_dim, norm_fn=norm_fn, num_layers=3)
            self.kernel_linear = MLP(channels, self.kernel_dim, norm_fn=norm_fn, num_layers=3)
            self.cls_linear = MLP(channels, self.instance_classes + 1, norm_fn=norm_fn, num_layers=2)

        self.init_weights()

        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()
        # if not self.semantic_only:
        #     for m in []:
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, return_loss=False):
        if return_loss:
            return self.forward_train(**batch)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                      semantic_labels, instance_labels, instance_pointnum, instance_cls,
                      pt_offset_labels, spatial_shape, instance_batch_idxs, batch_size, **kwargs):  
        losses = {}
        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        kernel_feats, mask_feats, output_feats = self.forward_backbone(input, v2p_map)


        # instance losses
        if not self.semantic_only:
            # acquire mask, cls, score
            # ins feat -> bs nProposals, nDim
            # cls_scores -> bs nProposals, nClasses
            cls_scores, instance_feats = self.get_instance_features(kernel_feats, output_feats, batch_idxs, batch_size)
            # mask -> bs * nPoints, nProposal
            mask_preds = self.get_mask_preds(instance_feats, mask_feats, batch_idxs)

            instance_loss = self.instance_loss_for_mask(mask_preds, cls_scores, 
                                                        instance_labels, 
                                                        instance_cls, batch_idxs, 
                                                        batch_size, instance_batch_idxs)
            losses.update(instance_loss)
        return self.parse_losses(losses)

    def point_wise_loss(self, semantic_scores, pt_offsets, semantic_labels, instance_labels,
                        pt_offset_labels):
        losses = {}
        semantic_loss = F.cross_entropy(
            semantic_scores, semantic_labels, ignore_index=self.ignore_label)
        losses['semantic_loss'] = semantic_loss

        pos_inds = instance_labels != self.ignore_label
        if pos_inds.sum() == 0:
            offset_loss = 0 * pt_offsets.sum()
        else:
            offset_loss = F.l1_loss(
                pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / pos_inds.sum()
        losses['offset_loss'] = offset_loss
        return losses
    
    @force_fp32(apply_to=('kernel_feats', 'output_feats'))
    def get_instance_features(self, kernel_feats, output_feats, batch_idxs, batch_size):
        """
        cls: bs, nProposals, nClasses
        ins_feats: bs * nProposals, nDim
        """
        instance_feats = []

        for i in range(batch_size):
            batch_id_mask = batch_idxs == i
            kernel_feats = kernel_feats * batch_id_mask.unsqueeze(1)
            i_kernel_feats = kernel_feats.masked_fill(~batch_id_mask.unsqueeze(1), 0.) # nPoints, nProposals
            i_output_feats = output_feats.masked_fill(~batch_id_mask.unsqueeze(1), 0.) # nPoints, nDim
            i_ins_feats = torch.mm(i_kernel_feats.transpose(1, 0), i_output_feats) # nProposals, nDim
            if self.norm_feat:
                i_ins_feats = F.normalize(i_ins_feats, p=2, dim=1)
            instance_feats.append(i_ins_feats)
        instance_feats = torch.cat(instance_feats, dim=0) # bs, nProposals, nDim
        cls_scores = self.cls_linear(instance_feats) 
        instance_feats = instance_feats.reshape(batch_size, self.kernel_dim, -1)
        cls_scores = cls_scores.reshape(batch_size, self.kernel_dim, -1)
        
        return cls_scores, instance_feats

    @force_fp32(apply_to=('instance_feats', 'mask_feats'))
    def get_mask_preds(self, instance_feats, mask_feats, batch_idxs):
        """
        instance_feats: bs, nProposals, nDim
        mask_feats: bs * nPoints, nDim
        mask_preds: bs * nPoints, nProposal
        """
        bs = instance_feats.shape[0]
        instance_feats_ = instance_feats.transpose(1, 2)
        mask_preds = mask_feats.new_full((mask_feats.shape[0], instance_feats.shape[1]), 0.)
        for i in range(bs):
            i_batch_idxs = batch_idxs == i
            i_mask_pred = torch.mm(mask_feats[i_batch_idxs, :], instance_feats_[i])
            mask_preds[i_batch_idxs, :] = i_mask_pred
        return mask_preds

    @force_fp32(apply_to=('mask_preds', 'cls_scores'))
    def instance_loss_for_mask(self, mask_preds, cls_scores, 
                               instance_labels, 
                               instance_cls, batch_idxs, 
                               batch_size, instance_batch_idxs):
        batch_pred_logits_list = []
        batch_pred_masks_list = []
        batch_gt_ins_labels_list = []
        batch_gt_masks_list = []

        # filter out background instances
        fg_inds = (instance_cls != self.ignore_label)
        fg_inds_ = torch.where(instance_cls != self.ignore_label)[0]
        fg_instance_cls = instance_cls[fg_inds]
        fg_instance_batch_idxs = instance_batch_idxs[fg_inds]
        fg_mask_gt = mask_preds.new_full((instance_labels.shape[0], fg_inds_.shape[0]), 0.)
        
        for i in range(fg_inds_.size(0)):
            fg_mask_gt[instance_labels == fg_inds_[i], i] = 1.

        for i in range(batch_size):
            i_indices_mask = batch_idxs == i
            i_ins_indices_mask = fg_instance_batch_idxs == i

            i_mask_pred = mask_preds[i_indices_mask, :]
            i_cls_pred = cls_scores[i]
            i_ins_cls = fg_instance_cls[i_ins_indices_mask]
            i_mask_gt = fg_mask_gt[i_indices_mask, :][:, i_ins_indices_mask]




            batch_pred_logits_list.append(i_cls_pred)
            batch_pred_masks_list.append(i_mask_pred)


            batch_gt_ins_labels_list.append(i_ins_cls)
            batch_gt_masks_list.append(i_mask_gt)
            assert i_ins_cls.shape[0] == i_mask_gt.shape[1]

        assert len(batch_pred_logits_list) == len(batch_gt_ins_labels_list)
        matcher_data = {
            "outputs": {
                "pred_logits" : batch_pred_logits_list,
                "pred_masks": batch_pred_masks_list
            },
            "targets": {
                "labels" : batch_gt_ins_labels_list,
                "masks" : batch_gt_masks_list
            }
        }

        # Hungarian Matching
        match_res = self.matcher(**matcher_data)
        
        
        # collect best matching pairs -> bs , n_points, n_ins_pair
        paired_pred_mask = [iy[:, ix] for (ix, _), iy in zip(match_res, batch_pred_masks_list)]
        paired_gt_mask = [iy[:, ix] for (_, ix), iy in zip(match_res, batch_gt_masks_list)]

        paired_pred_cls = [iy[ix] for (ix, _), iy in zip(match_res, batch_pred_logits_list)]
        paired_gt_cls = [iy[ix] for (_, ix), iy in zip(match_res, batch_gt_ins_labels_list)]
        paired_pred_cls = torch.cat(paired_pred_cls)
        paired_gt_cls = torch.cat(paired_gt_cls)

        losses = {}
        # mask loss  (dice loss + focal loss are used in MaskFormer)
        focal_loss = 0.
        # ce_loss = 0.
        d_loss = 0.
        for i in range(batch_size):
            # ce_loss += F.binary_cross_entropy_with_logits(paired_pred_mask[i].sigmoid(), paired_gt_mask[i])
            focal_loss += sigmoid_focal_loss(paired_pred_mask[i], paired_gt_mask[i], paired_gt_mask[i].shape[1])
            d_loss += dice_loss(paired_pred_mask[i].transpose(0,1), paired_gt_mask[i].transpose(0,1)) / paired_gt_mask[i].shape[1]
        # ce_loss /= batch_size
        d_loss /= batch_size
        focal_loss /= batch_size

        # cls loss (cross entropy loss is used in MaskFormer)
        cls_loss = F.cross_entropy(paired_pred_cls, paired_gt_cls, ignore_index=self.ignore_label)

        # losses['ce_loss'] = ce_loss
        losses['d_loss'] = d_loss
        losses['focal_loss'] = focal_loss
        losses['cls_loss'] = cls_loss
        return losses


    @force_fp32(apply_to=('cls_scores', 'mask_scores', 'iou_scores'))
    def instance_loss(self, cls_scores, mask_scores, iou_scores, proposals_idx, proposals_offset,
                      instance_labels, instance_pointnum, instance_cls, instance_batch_idxs):
        losses = {}
        proposals_idx = proposals_idx[:, 1].cuda()
        proposals_offset = proposals_offset.cuda()

        # cal iou of clustered instance
        ious_on_cluster = get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                                  instance_pointnum)

        # filter out background instances
        fg_inds = (instance_cls != self.ignore_label)
        fg_instance_cls = instance_cls[fg_inds]
        fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

        # overlap > thr on fg instances are positive samples
        max_iou, gt_inds = fg_ious_on_cluster.max(1)
        pos_inds = max_iou >= self.train_cfg.pos_iou_thr
        pos_gt_inds = gt_inds[pos_inds]

        # compute cls loss. follow detection convention: 0 -> K - 1 are fg, K is bg
        labels = fg_instance_cls.new_full((fg_ious_on_cluster.size(0), ), self.instance_classes)
        labels[pos_inds] = fg_instance_cls[pos_gt_inds]
        cls_loss = F.cross_entropy(cls_scores, labels)
        losses['cls_loss'] = cls_loss

        # compute mask loss
        mask_cls_label = labels[instance_batch_idxs.long()]
        slice_inds = torch.arange(
            0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
        mask_scores_sigmoid_slice = mask_scores.sigmoid()[slice_inds, mask_cls_label]
        mask_label = get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
                                    instance_pointnum, ious_on_cluster, self.train_cfg.pos_iou_thr)
        mask_label_weight = (mask_label != -1).float()
        mask_label[mask_label == -1.] = 0.5  # any value is ok
        mask_loss = F.binary_cross_entropy(
            mask_scores_sigmoid_slice, mask_label, weight=mask_label_weight, reduction='sum')
        mask_loss /= (mask_label_weight.sum() + 1)
        losses['mask_loss'] = mask_loss

        # compute iou score loss
        ious = get_mask_iou_on_pred(proposals_idx, proposals_offset, instance_labels,
                                    instance_pointnum, mask_scores_sigmoid_slice.detach())
        fg_ious = ious[:, fg_inds]
        gt_ious, _ = fg_ious.max(1)
        slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
        iou_score_weight = (labels < self.instance_classes).float()
        iou_score_slice = iou_scores[slice_inds, labels]
        iou_score_loss = F.mse_loss(iou_score_slice, gt_ious, reduction='none')
        iou_score_loss = (iou_score_loss * iou_score_weight).sum() / (iou_score_weight.sum() + 1)
        losses['iou_score_loss'] = iou_score_loss
        return losses

    def parse_losses(self, losses):
        loss = sum(v for v in losses.values())
        losses['loss'] = loss
        for loss_name, loss_value in losses.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses[loss_name] = loss_value.item()
        return loss, losses

    @cuda_cast
    def forward_test(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                     semantic_labels, instance_labels, pt_offset_labels, spatial_shape, batch_size,
                     scan_ids, **kwargs):
        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        kernel_feats, mask_feats, output_feats = self.forward_backbone(
            input, v2p_map, x4_split=self.test_cfg.x4_split)
        if self.test_cfg.x4_split:
            coords_float = self.merge_4_parts(coords_float)
            # semantic_labels = self.merge_4_parts(semantic_labels)
            instance_labels = self.merge_4_parts(instance_labels)
            # pt_offset_labels = self.merge_4_parts(pt_offset_labels)
        # semantic_preds = semantic_scores.max(1)[1]
        ret = dict(
            scan_id=scan_ids[0],
            coords_float=coords_float.cpu().numpy(),
            semantic_labels=semantic_labels.cpu().numpy(),
            offset_labels=pt_offset_labels.cpu().numpy(),
            instance_labels=instance_labels.cpu().numpy())
        if not self.semantic_only:
            # pred_instances
            cls_scores, instance_feats = self.get_instance_features(kernel_feats, output_feats, batch_idxs, batch_size)
            
            mask_preds = self.get_mask_preds(instance_feats, mask_feats, batch_idxs)
            
            pred_instances = self.get_instances_from_masks(scan_ids[0], mask_preds, cls_scores)
            gt_instances = self.get_gt_instances(semantic_labels, instance_labels)
            ret.update(dict(pred_instances=pred_instances, gt_instances=gt_instances))
        return ret

    def forward_backbone(self, input, input_map, x4_split=False):
        if x4_split:
            output_feats = self.forward_4_parts(input, input_map)
            output_feats = self.merge_4_parts(output_feats)
        else:
            output = self.input_conv(input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()] # nPoints, nDim
        

        if not self.semantic_only:
            mask_feats = self.mask_linear(output_feats)
            kernel_feats = self.kernel_linear(output_feats) # nPoints, nPro

        return kernel_feats, mask_feats, output_feats 

    def forward_4_parts(self, x, input_map):
        """Helper function for s3dis: devide and forward 4 parts of a scene."""
        outs = []
        for i in range(4):
            inds = x.indices[:, 0] == i
            feats = x.features[inds]
            coords = x.indices[inds]
            coords[:, 0] = 0
            x_new = spconv.SparseConvTensor(
                indices=coords, features=feats, spatial_shape=x.spatial_shape, batch_size=1)
            out = self.input_conv(x_new)
            out = self.unet(out)
            out = self.output_layer(out)
            outs.append(out.features)
        outs = torch.cat(outs, dim=0)
        return outs[input_map.long()]

    def merge_4_parts(self, x):
        """Helper function for s3dis: take output of 4 parts and merge them."""
        inds = torch.arange(x.size(0), device=x.device)
        p1 = inds[::4]
        p2 = inds[1::4]
        p3 = inds[2::4]
        p4 = inds[3::4]
        ps = [p1, p2, p3, p4]
        x_split = torch.split(x, [p.size(0) for p in ps])
        x_new = torch.zeros_like(x)
        for i, p in enumerate(ps):
            x_new[p] = x_split[i]
        return x_new

    @force_fp32(apply_to=('semantic_labels'))
    def get_instances_from_gt(self, scan_id, semantic_labels, instance_labels):
        pts = semantic_labels.shape[0]
        ins_cls_gt = semantic_labels - self.semantic_classes + self.instance_classes
        ins_cls_gt[ins_cls_gt < 0] = 0
        ins_label_unique = instance_labels.unique()
        mask_gt = []
        cls_gt = []
        ins_num = 0
        for i in range(ins_label_unique.shape[0]):
            if ins_label_unique[i] != self.ignore_label:
                mask_gt.append(instance_labels == ins_label_unique[i])
                cls_gt.append(semantic_labels[instance_labels == ins_label_unique[i]][0].item())
                ins_num += 1

        mask_gt = torch.stack(mask_gt, dim=1).int().cuda()
        cls_gt = torch.Tensor(cls_gt).long().cuda()
        score_gt = torch.ones_like(cls_gt, dtype=torch.float)

        # filter floor & wall
        label_shift = self.semantic_classes - self.instance_classes
        cls_gt = cls_gt - label_shift + 1
        inds = cls_gt >= 1
        cls_gt = cls_gt[inds]
        score_gt = score_gt[inds]
        mask_gt = mask_gt[:, inds]

        cls_gt = cls_gt.cpu().numpy()
        score_gt = score_gt.cpu().numpy()
        mask_gt = mask_gt.cpu().numpy()

        instances = []
        for i in range(cls_gt.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_gt[i]
            pred['conf'] = score_gt[i]
            pred['pred_mask'] = rle_encode(mask_gt[:, i])
            instances.append(pred)
        return instances

    @force_fp32(apply_to=('mask_preds', 'cls_scores'))
    def get_instances_from_masks(self, scan_id, mask_preds, cls_scores):
        
        mask_preds = torch.where(mask_preds.sigmoid() > self.test_cfg.mask_score_thr, 1, 0).int()
        
        score_map = cls_scores.squeeze(0).softmax(-1)

        score_pred, cls_pred = score_map.max(-1)
        
        # filter 1 score
        inds = score_pred > self.test_cfg.cls_score_thr
        cls_pred = cls_pred[inds]
        score_pred = score_pred[inds]
        mask_preds = mask_preds[:, inds]

        # filter 2 num of points
        npoint = mask_preds.sum(0)
        inds = npoint >= self.test_cfg.min_npoint
        cls_pred = cls_pred[inds]
        score_pred = score_pred[inds]
        mask_preds = mask_preds[:, inds]

        # sort
        _, inds = cls_pred.sort(0, descending=True)
        cls_pred = cls_pred[inds]
        score_pred = score_pred[inds]
        mask_preds = mask_preds[:, inds]
        

        cls_pred = cls_pred.cpu().numpy()
        score_pred = score_pred.cpu().numpy()
        mask_preds = mask_preds.cpu().numpy()

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_pred[i] + 1
            pred['conf'] = score_pred[i]
            pred['pred_mask'] = rle_encode(mask_preds[:, i])
            instances.append(pred)
        return instances


    @force_fp32(apply_to=('semantic_scores', 'cls_scores', 'iou_scores', 'mask_scores'))
    def get_instances(self, scan_id, proposals_idx, semantic_scores, cls_scores, iou_scores,
                      mask_scores):
        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        semantic_pred = semantic_scores.max(1)[1]
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []
        for i in range(self.instance_classes):
            if i in self.sem2ins_classes:
                cls_pred = cls_scores.new_tensor([i + 1], dtype=torch.long)
                score_pred = cls_scores.new_tensor([1.], dtype=torch.float32)
                mask_pred = (semantic_pred == i)[None, :].int()
            else:
                cls_pred = cls_scores.new_full((num_instances, ), i + 1, dtype=torch.long)
                cur_cls_scores = cls_scores[:, i]
                cur_iou_scores = iou_scores[:, i]
                cur_mask_scores = mask_scores[:, i]
                score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
                mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')
                mask_inds = cur_mask_scores > self.test_cfg.mask_score_thr
                cur_proposals_idx = proposals_idx[mask_inds].long()
                mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1

                # filter low score instance
                inds = cur_cls_scores > self.test_cfg.cls_score_thr
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]

                # filter too small instances
                npoint = mask_pred.sum(1)
                inds = npoint >= self.test_cfg.min_npoint
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]
            cls_pred_list.append(cls_pred)
            score_pred_list.append(score_pred)
            mask_pred_list.append(mask_pred)
        cls_pred = torch.cat(cls_pred_list).cpu().numpy()
        score_pred = torch.cat(score_pred_list).cpu().numpy()
        mask_pred = torch.cat(mask_pred_list).cpu().numpy()

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            instances.append(pred)
        return instances

    def get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = self.semantic_classes - self.instance_classes
        semantic_labels = semantic_labels - label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins.cpu().numpy()
        return gt_ins

    @force_fp32(apply_to='feats')
    def clusters_voxelization(self,
                              clusters_idx,
                              clusters_offset,
                              feats,
                              coords,
                              scale,
                              spatial_shape,
                              rand_quantize=False):
        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = sec_min(coords, clusters_offset.cuda())
        coords_max = sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = voxelization(feats, out_map.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats,
                                                     out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)
        return voxelization_feats, inp_map

    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    @force_fp32(apply_to=('x'))
    def global_pool(self, x, expand=False):
        indices = x.indices[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1, ), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool

        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x
