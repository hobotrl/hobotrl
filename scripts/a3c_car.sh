#!/usr/bin/env bash
#
#	usage: . ./scripts/a3c_car.sh
#

. ./scripts/cluster.sh ./test/exp_car.py A3CCarDiscrete2 1 1 --log_dir ./log/CarRacing_A3C --start_port 2345 --start_device 0