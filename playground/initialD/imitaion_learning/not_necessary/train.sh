#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
train_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_fenkai_rm_stp/train"
val_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_fenkai_rm_stp/val"
#train_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/train"
#val_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/val"
log_dir="./log_sl_rnd_v2"
mkdir -p $log_dir
unbuffer python sl_train_v2.py  \
    --train_dir $train_dir \
    --val_dir $val_dir \
    --batch_size 128 \
    --train_interval 100 \
    --val_interval 100 \
    --val_iter 10 \
    --l2 0.0001 \
    --initial_lr 0.01 \o
    --max_step 100000 \
    --gpu_fraction 0.7 \
    --log_dir $log_dir \
    --stack_num 3 \
    > $log_dir/worker0.txt 2>&1 &

