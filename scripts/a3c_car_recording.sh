#!/usr/bin/env bash
#
#	usage: . ./scripts/a3c_car.sh
#

exp_file=/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/exp_car_racing.py
exp_name=A3CCarRecordingDiscrete2
log_dir=/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/log/$exp_name
save_dir=/home/pirate03/hobotrl_data/A3CCarRecordingDiscrete2
cluster="{'ps':['localhost:2252'], 'worker':['localhost:2253', 'localhost:2254']}"
#cluster="{'ps':['localhost:2242'], 'worker':['localhost:2243','localhost:2244','localhost:2245','localhost:2246']}"
worker_n=2
device_n=2
mkdir -p $log_dir
python $exp_file run --name $exp_name --cluster "$cluster" --job ps --logdir $log_dir > $log_dir/ps.txt 2>&1 &
end_i=$(expr $worker_n - 1)
echo "job $exp_name with worker: [ 0 .. $end_i ]"
for i in $(seq 0 $end_i)
do
	device=$(expr $i % $device_n)
	CUDA_VISIBLE_DEVICES=$device python $exp_file run --name $exp_name --cluster "$cluster" --job worker --index $i --logdir $log_dir --savedir $save_dir > $log_dir/worker.$i.txt 2>&1 &
done

