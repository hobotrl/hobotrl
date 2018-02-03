#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
#exp_name3="Freeway_mom_I2A"
#. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name3} 1 1 --log_dir ~/hobotrl/log/Freeway/${exp_name3} --start_port 2325 --start_device 3

#log_dir="./log/Freeway/FreewayOTDQN_mom"
#mkdir -p ${log_dir}
#CUDA_VISIBLE_DEVICES=1 python test/exp_freeway.py run --name FreewayOTDQN_mom --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &
#
#log_dir1="./log/Freeway/FreewayOTDQN_state_256"
#mkdir -p ${log_dir1}
#CUDA_VISIBLE_DEVICES=2 python test/exp_freeway.py run --name FreewayOTDQN_state_256 --logdir ${log_dir1} > ${log_dir1}/log.txt 2>&1 &

log_dir="./log/Freeway/Freeway_ob_I2A_decoder"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=0 python test/exp_freeway.py run --name Freeway_ob_I2A_decoder --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &