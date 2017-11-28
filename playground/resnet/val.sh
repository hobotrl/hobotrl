#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
val_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_obj80_vec_rewards_docker005_no_early_stopping_all_green/valid"
log_dir="./val_ckp"
mkdir -p $log_dir
unbuffer python val_im_model.py  \
    --val_dir $val_dir \
    --log_dir $log_dir \
#    > $log_dir/worker0.txt 2>&1 &


# ResNet-18 baseline loaded from torch resnet-18.t7
# Finetune for 10 epochs
