#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29501}

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py --dist $CONFIG ${@:3} 
# --resume ./work_dirs/mask3dv4_scannet_train6/latest.pth
