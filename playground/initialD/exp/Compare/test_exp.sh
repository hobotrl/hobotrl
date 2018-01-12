#!/usr/bin/env bash
gpu_n=0
exp_script="./exp_DrSimKub_AsyncDQN_LanePaper_ExpID01.py"
log_dir="../../../../../../work/agents/exp01/"
CUDA_VISIBLE_DEVICES=$gpu_n unbuffer python $exp_script --test True