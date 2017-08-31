#!/bin/sh
log_dir="./log"
mkdir -p $log_dir
python exp_DrSim_dagger_oo.py | tee $log_dir/worker0.txt &