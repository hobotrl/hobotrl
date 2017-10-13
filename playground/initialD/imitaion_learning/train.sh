#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
#train_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_fenkai_rm_stp/train"
#val_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_fenkai_rm_stp/val"
train_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/train1"
val_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/val1"
log_dir="./log_sl_rnd_test_simple_stack1"
mkdir -p $log_dir
unbuffer python sl_train_v2.py  \
    --train_dir $train_dir \
    --val_dir $val_dir \
    --batch_size 180 \
    --train_interval 20 \
    --val_interval 20 \
    --val_iter 5 \
    --l2 0.001 \
    --initial_lr 0.001 \
    --max_step 200 \
    --log_dir $log_dir \
#    > $log_dir/worker0.txt 2>&1 &

