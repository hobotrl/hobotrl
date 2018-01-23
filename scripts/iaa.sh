#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
#exp_name1="Freeway_A3C_half_1e_4"
#. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name1} 4 2 --log_dir ~/hobotrl/log/Freeway/${exp_name1} --start_port 2345 --start_device 2

log_dir="./log/MsPacman/MsPacmanOTDQN"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=1 python test/exp_iaa.py run --name MsPacmanOTDQN --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &

#log_dir1="./log/MsPacman/OTDQN_ob_new"
#mkdir -p ${log_dir1}
#CUDA_VISIBLE_DEVICES=1 python test/exp_iaa.py run --name OTDQN_ob_MsPacman --logdir ${log_dir1} > ${log_dir1}/log.txt 2>&1 &