#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#

log_dir="./log/Model"
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=2 python test/exp_model.py run --name Model --logdir ${log_dir} > ${log_dir}/log.txt 2>&1 &