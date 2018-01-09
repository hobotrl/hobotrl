#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
exp_name1="Freeway_A3C_half"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name1} 4 2 --log_dir ~/hobotrl/log/Freeway/${exp_name1} --start_port 2315 --start_device 2
