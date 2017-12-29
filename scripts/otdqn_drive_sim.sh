#!/usr/bin/env bash
#
#	usage: . ./scripts/otdqn_drive_sim.sh
#
log_dir="./log/Simul/OTDQN"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=0 python test/exp_car_flow.py run --name OTDQNModelDriving --render_once false --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &