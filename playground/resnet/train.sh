#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
#export LD_PRELOAD="/usr/lib/libtcmalloc.so"
#train_dir="./resnet_baseline"
train_dataset="/home/pirate03/PycharmProjects/hobotrl/data/records_v1/filter_action3/train.tfrecords"
#train_image_root="/data1/common_datasets/imagenet_resized/"
val_dataset="/home/pirate03/PycharmProjects/hobotrl/data/records_v1/filter_action3/val.tfrecords"
#val_image_root="/  data1/common_datasets/imagenet_resized/ILSVRC2012_val/"
log_dir="./log3_tmp"
train_dir=$log_dir
mkdir -p $log_dir
unbuffer python train.py  \
    --train_dataset $train_dataset \
    --val_dataset $val_dataset \
    --batch_size 256 \
    --num_gpus 1 \
    --val_interval 200 \
    --val_iter 20 \
    --l2_weight 0.001 \
    --initial_lr 0.01 \
    --lr_step_epoch 10.0,20.0 \
    --lr_decay 0.1 \
    --max_steps 10100 \
    --train_dir $train_dir \
    --checkpoint_interval 1000 \
    --gpu_fraction 0.7 \
    --display 100 \
    --finetune True \
    --checkpoint "/home/pirate03/PycharmProjects/resnet-18-tensorflow/log2_15/model.ckpt-9999" \
    > $log_dir/worker0.txt 2>&1 &


# ResNet-18 baseline loaded from torch resnet-18.t7
# Finetune for 10 epochs
