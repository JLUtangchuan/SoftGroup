#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   batch_vis.py
@Time    :   2022/06/22 09:58:22
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   批量生成可视化结果
'''
import sys
import os
from tqdm import tqdm


tasks = ["input", "instance_gt", "instance_pred"]
dataset = "scannetv2"
prediction_path = "./exp/vis/maskv5"
out_path = "./exp/vis/maskv5/out"

def get_rooms(prediction_path, max_num = 50):
    _dir = os.path.join(prediction_path, "gt_instance")
    # load file names
    files = os.listdir(_dir)
    # base name os.path.basename
    for i in range(len(files)):
        files[i] = os.path.splitext(files[i])[0]
    return files[:max_num]

if __name__ == '__main__':
    rooms = get_rooms(prediction_path)
    # 执行 shell 命令
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for r in tqdm(rooms):
        for t in tasks:
            out = os.path.join(out_path, "val_{r}_{t}.ply".format(r=r, t=t))
            cmd = "python tools/visualization.py --dataset {} --task {} --prediction_path {} --out {} --room_name {}".format(dataset, t, prediction_path, out, r)
            os.system(cmd)

# 
# python visualization.py --dataset scannetv2 --prediction_path /home/tc/workdir/SoftGroupV2/exp/work_dirs/v2_result --task instance_pred --out /home/tc/workdir/SoftGroupV2/exp/work_dirs/v2_result/out/val_scene0030_00_instance_pred.ply --room_name scene0030_00

# input/semantic_gt/semantic_pred/offset_semantic_pred/instance_gt/instance_pred