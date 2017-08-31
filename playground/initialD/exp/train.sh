#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1
train_dir="./log_check_noval_scene"
mkdir -p $train_dir
unbuffer python exp_DrSim_dagger_oo.py \
    --train_dir $train_dir \
    --checkpoint "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/exp/log/model.ckpt-10" \
    | tee $train_dir/log_check_noval_scene.txt &

#python exp_DrSim_dagger_oo.py --train_dir "./log_check_noval_scene" checkpoint \
#"/home/pirate03/PycharmProjects/hobotrl/playground/initialD/exp/log/model.ckpt-10" \
#| tee "./log_check_noval_scene/log_check_noval_scene.txt"