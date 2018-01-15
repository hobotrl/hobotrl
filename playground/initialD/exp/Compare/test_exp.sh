#!/usr/bin/env bash
gpu_n=$1
script=$2
log_fdir=$3
# ../../../../../../work/agents/exp01/
start=$4
end=$5
for number in `seq $start 1 $end`
do
    log_dir=${log_fdir}${number}/
#    echo "CUDA_VISIBLE_DEVICES=$gpu_n unbuffer python $script --test True --logdir $log_dir > ${log_dir}test.txt 2>&1 &"
    CUDA_VISIBLE_DEVICES=$gpu_n unbuffer python $script --test True --dir_prefix $log_dir > ${log_dir}test.txt 2>&1 &
done