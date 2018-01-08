#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
exp_name1="Freeway_A3C_halfE2"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name1} 2 1 --log_dir ~/hobotrl/log/Freeway/${exp_name1} --start_port 2305 --start_device 2

exp_name2="Freeway_A3C_half"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name2} 2 1 --log_dir ~/hobotrl/log/Freeway/${exp_name2} --start_port 2315 --start_device 2

exp_name3="Freeway_A3C_halfE1"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name3} 2 1 --log_dir ~/hobotrl/log/Freeway/${exp_name3} --start_port 2325 --start_device 3

exp_name4="Freeway_A3C_halfLR1"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name4} 2 1 --log_dir ~/hobotrl/log/Freeway/${exp_name4} --start_port 2335 --start_device 3

exp_name5="Freeway_A3C_halfLR2"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name5} 2 1 --log_dir ~/hobotrl/log/Freeway/${exp_name5} --start_port 2345 --start_device 3

exp_name6="Freeway_A3C_half"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name6} 4 1 --log_dir ~/hobotrl/log/Freeway/${exp_name6}"4" --start_port 2355 --start_device 0

exp_name7="Freeway_A3C_half"
. ./scripts/cluster.sh ./test/exp_freeway.py ${exp_name7} 8 1 --log_dir ~/hobotrl/log/Freeway/${exp_name7}"8" --start_port 2365 --start_device 1
