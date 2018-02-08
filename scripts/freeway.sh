#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
#exp_name3="Freeway_ob_I2A"
#. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name3} 4 2 --log_dir ~/hobotrl/log/Freeway/${exp_name3} --start_port 2325 --start_device 1

log_dir="./log/Freeway/FreewayOTDQN_mom"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=2 python test/exp_freeway.py run --name FreewayOTDQN_mom --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &

#log_dir1="./log/Freeway/FreewayOTDQN_mom_1600"
#mkdir -p ${log_dir1}
#CUDA_VISIBLE_DEVICES=2 python test/exp_freeway.py run --name FreewayOTDQN_mom_1600 --logdir ${log_dir1} > ${log_dir1}/log.txt 2>&1 &

#log_dir="./log/Freeway/FreewayOTDQN_goal_256"
#mkdir -p ${log_dir}
#CUDA_VISIBLE_DEVICES=3 python test/exp_freeway.py run --name FreewayOTDQN_goal_256 --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &

#log_dir="./log/Freeway/FreewayOTDQN_goal"
#mkdir -p ${log_dir}
#CUDA_VISIBLE_DEVICES=3 python test/exp_freeway.py run --name FreewayOTDQN_goal --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &
