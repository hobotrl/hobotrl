#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
exp_name1="Freeway_A3C_half_3e_4"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name1} 4 1 --log_dir ~/hobotrl/log/Freeway/${exp_name1} --start_port 2345 --start_device 0

#log_dir="./log/Freeway/otdqn"
#mkdir -p ${log_dir}
#CUDA_VISIBLE_DEVICES=3 python test/exp_freeway.py run --name otdqn --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &