#!/usr/bin/env bash
#
#	usage: . ./scripts/otdqn_drive_sim.sh
#
CUDA_VISIBLE_DEVICES=0 python test/exp_deeprl.py run --name ACConPendulumSearch
# --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &