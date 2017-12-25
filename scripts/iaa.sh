#!/usr/bin/env bash
#
#	usage: . ./scripts/iaa.sh
#
. ./scripts/cluster.sh ./test/exp_car_flow.py OTDQNModelDriving 1 1 --log_dir ./log/OTDQNModelDriving --start_port 2345 --start_device 0
