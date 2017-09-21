#
#	usage: . ./scripts/icm_cartpole.sh
#

exp_file=./test/exp_icm.py
exp_name=ICMLinear
log_dir=./log/MountainCar/1-0.05-1
# cluster="{'ps':['localhost:2232'], 'worker':['localhost:2233', 'localhost:2234', 'localhost:2235']}"
cluster="{'ps':['localhost:2242'], 'worker':['localhost:2243', 'localhost:2244', 'localhost:2245', 'localhost:2246']}"
worker_n=4
device_n=4
mkdir -p $log_dir
python $exp_file run --name $exp_name --cluster "$cluster" --job ps --logdir $log_dir > $log_dir/ps.txt 2>&1 &
end_i=$(expr $worker_n - 1)
echo "job $exp_name with worker: [ 0 .. $end_i ]"
for i in $(seq 0 $end_i)
do
	device=$(expr $i % $device_n)
	CUDA_VISIBLE_DEVICES=$device python $exp_file run --name $exp_name --cluster "$cluster" --job worker --index $i --logdir $log_dir > $log_dir/worker.$i.txt 2>&1 &
done
