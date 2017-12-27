#!/usr/bin/env bash
#
#	usage: . ./scripts/otdqn_drive_sim.sh
#
mkdir -p ./log/Simul/OTDQN
CUDA_VISIBLE_DEVICES=0 python test/exp_car_flow.py run --name OTDQNModelDriving --render_once false --logdir ./log/Simul/OTDQN > ./log/Simul/OTDQN/log.txt 2>&1 &