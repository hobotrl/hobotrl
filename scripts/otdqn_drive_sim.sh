#!/usr/bin/env bash
#
#	usage: . ./scripts/otdqn_drive_sim.sh
#
log_dir="./log/CarRacing/OTDQN_mom_decoder"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=2 python test/exp_car_flow.py run --name OTDQNModelCar_mom_decoder --render_once true --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &

#log_dir1="./log/CarRacing/OTDQN_mom"
#mkdir -p ${log_dir1}
#CUDA_VISIBLE_DEVICES=0 python test/exp_car_flow.py run --name OTDQNModelCar --render_once true --logdir ${log_dir1} > ${log_dir1}/log.txt 2>&1 &