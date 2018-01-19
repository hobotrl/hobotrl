#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
#exp_name1="Freeway_A3C_half_1e_4"
#. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name1} 4 2 --log_dir ~/hobotrl/log/Freeway/${exp_name1} --start_port 2345 --start_device 2

log_dir="./log/MsPacman/OTDQN_mom_decoder"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=0 python test/exp_iaa.py run --name MsPacmanOTDQN_mom_decoder --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &

log_dir1="./log/MsPacman/OTDQN_ob"
mkdir -p ${log_dir1}
CUDA_VISIBLE_DEVICES=1 python test/exp_iaa.py run --name OTDQN_ob_MsPacman --render_once true --logdir ${log_dir1} > ${log_dir1}/log.txt 2>&1 &

#log_dir2="./log/MsPacman/OTDQN_mom_1600"
#mkdir -p ${log_dir2}
#CUDA_VISIBLE_DEVICES=3 python test/exp_iaa.py run --name MsPacmanOTDQN_mom_1600 --logdir ${log_dir2} > ${log_dir2}/log.txt 2>&1 &
#
#log_dir3="./log/MsPacman/OTDQN_mom"
#mkdir -p ${log_dir3}
#CUDA_VISIBLE_DEVICES=2 python test/exp_iaa.py run --name MsPacmanOTDQN --logdir ${log_dir3} > ${log_dir3}/log.txt 2>&1 &