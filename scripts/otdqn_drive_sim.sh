#!/usr/bin/env bash
#
#	usage: . ./scripts/otdqn_drive_sim.sh
#
log_dir="./log/CarRacing/OTDQN_mom_1600"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=3 python test/exp_car_flow.py run --name OTDQNModelCar_mom_1600 --render_once true --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &

#log_dir="./log/CarRacing/OTDQN_ob"
#mkdir -p ${log_dir}
#CUDA_VISIBLE_DEVICES=1 python test/exp_car_flow.py run --name OTDQN_ob --render_once true --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &