#!/usr/bin/env bash
#
#	usage: . ./scripts/otdqn_drive_sim.sh
#
CUDA_VISIBLE_DEVICES=0 python test/exp_freeway.py run --name Freeway_search
# --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &