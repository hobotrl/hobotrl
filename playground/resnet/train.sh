#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
#export LD_PRELOAD="/usr/lib/libtcmalloc.so"
#train_dir="./resnet_baseline"
train_dir="/home/pirate03/hobotrl_data/A3CCarRecordingDiscrete2/train"
#train_image_root="/data1/common_datasets/imagenet_resized/"
val_dir="/home/pirate03/hobotrl_data/A3CCarRecordingDiscrete2/valid"
#val_image_root="/  data1/common_datasets/imagenet_resized/ILSVRC2012_val/"
log_dir="./A3CCarracing_pi"
mkdir -p $log_dir
unbuffer python train.py  \
    --train_dir $train_dir \
    --val_dir $val_dir \
    --batch_size 128 \
    --num_gpus 1 \
    --val_interval 200 \
    --val_iter 5 \
    --l2_weight 0.001 \
    --initial_lr 0.01 \
    --lr_step_epoch 10.0,20.0 \
    --lr_decay 0.01 \
    --max_steps 50000 \
    --log_dir $log_dir \
    --checkpoint_interval 5000 \
    --gpu_fraction 0.4 \
    --display 100 \
    > $log_dir/worker0.txt 2>&1 &


# ResNet-18 baseline loaded from torch resnet-18.t7
# Finetune for 10 epochs
