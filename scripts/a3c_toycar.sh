#!/usr/bin/env bash
#
#	usage: . ./scripts/a3c_concar.sh
#

. ./scripts/cluster.sh ./test/exp_car.py A3CToyCarDiscrete 2 2 "--render_once false --render_interval 1"
# . ./scripts/cluster.sh ./test/exp_car.py A3CToyCarDiscrete 4 4 "--render_once true"


