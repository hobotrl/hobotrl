#!/usr/bin/env bash
#
#	usage: . ./scripts/otdqn_drive_sim.sh
#
log_dir="./log/CarRacing/OTDQN"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=3 python test/exp_car_flow.py run --name OTDQNModelCar --render_once true --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &