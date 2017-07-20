#
#	usage: . ./scripts/a3c_pong.sh
#

exp_file=./test/exp_car.py
exp_name=A3CCarDiscrete
log_dir=./log/$exp_name
device=3
# cluster="{'ps':['localhost:2232'], 'worker':['localhost:2233', 'localhost:2234', 'localhost:2235']}"
cluster="{'ps':['localhost:2242'], 'worker':['localhost:2243']}"
worker_n=1

mkdir -p $log_dir 
python $exp_file run --name $exp_name --cluster "$cluster" --job ps > $log_dir/ps.txt 2>&1 &
end_i=$(expr $worker_n - 1)
echo "job $exp_name with worker: [ 0 .. $end_i ]"
for i in $(seq 0 $end_i)
do
	CUDA_VISIBLE_DEVICES=$device python $exp_file run --name $exp_name --cluster "$cluster" --job worker --index $i > $log_dir/worker.$i.txt 2>&1 &
done
