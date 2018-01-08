#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
exp_name="Freeway_A3C_half"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name} 2 1 --log_dir ~/hobotrl/log/Freeway/${exp_name} --start_port 2365 --start_device 3
