#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#

log_dir="./log/Freeway_old/FreewayOTDQN_mom"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=3 python test/exp_model.py run --name Model --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &