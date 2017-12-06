#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
. ./scripts/cluster.sh ./test/exp_iaa.py I2A 4 2 --log_dir ./log/I2AMsPacman --start_port 2345 --start_device 0
