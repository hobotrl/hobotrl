#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
#exp_name3="Freeway_A3C_halfE1"
#. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name3} 2 1 --log_dir ~/hobotrl/log/Freeway/${exp_name3} --start_port 2325 --start_device 3

log_dir="./log/Freeway/OTDQN_mom"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=3 python test/exp_freeway.py run --name OTDQNFreeway --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &

#log_dir="./log/Freeway/OTDQN_ob"
#mkdir -p ${log_dir}
#CUDA_VISIBLE_DEVICES=1 python test/exp_freeway.py run --name OTDQN_ob_freeway --render_once true --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &