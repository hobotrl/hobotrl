#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
exp_name1="Freeway_mom_half"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name1} 2 2 --log_dir ~/hobotrl/log/Freeway/${exp_name1} --start_port 2305 --start_device 0

#exp_name3="Freeway_mom_I2A_half"
#. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name3} 2 1 --log_dir ~/hobotrl/log/Freeway/${exp_name3} --start_port 2325 --start_device 3
