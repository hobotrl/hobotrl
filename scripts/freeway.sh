#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
exp_name="Freeway_mom_I2A"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name} 1 1 --log_dir ./log/Freeway/${exp_name} --start_port 2345 --start_device 0
