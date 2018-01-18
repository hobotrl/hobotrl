#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
#exp_name3="Freeway_mom_I2A"
#. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name3} 1 1 --log_dir ~/hobotrl/log/Freeway/${exp_name3} --start_port 2325 --start_device 3

log_dir="./log/Freeway/OTDQN_mom_1600"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=0 python test/exp_freeway.py run --name OTDQN_mom_1600 --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &

#log_dir="./log/Freeway/OTDQN_ob"
#mkdir -p ${log_dir}
#CUDA_VISIBLE_DEVICES=1 python test/exp_freeway.py run --name OTDQN_ob_Freeway --render_once true --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &