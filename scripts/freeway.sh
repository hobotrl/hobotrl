#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
#exp_name3="Freeway_mom_I2A"
#. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name3} 1 1 --log_dir ~/hobotrl/log/Freeway/${exp_name3} --start_port 2325 --start_device 3

log_dir="./log/Freeway/OTDQN_mom_decoder"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=2 python test/exp_freeway.py run --name FreewayOTDQN_mom_decoder --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &

log_dir1="./log/Freeway/OTDQN_ob"
mkdir -p ${log_dir1}
CUDA_VISIBLE_DEVICES=1 python test/exp_freeway.py run --name OTDQN_ob_Freeway --render_once true --logdir ${log_dir1} > ${log_dir1}/log.txt 2>&1 &

log_dir2="./log/Freeway/OTDQN_mom_1600"
mkdir -p ${log_dir2}
CUDA_VISIBLE_DEVICES=3 python test/exp_freeway.py run --name FreewayOTDQN_mom_1600 --logdir ${log_dir2} > ${log_dir2}/log.txt 2>&1 &

log_dir3="./log/Freeway/OTDQN_mom"
mkdir -p ${log_dir3}
CUDA_VISIBLE_DEVICES=0 python test/exp_freeway.py run --name FreewayOTDQN_mom --logdir ${log_dir3} > ${log_dir3}/log.txt 2>&1 &