#!/usr/bin/env bash
#
#   usage: . ./scripts/cluster.sh exp_file exp_name [worker_n [device_n]]"
#

print_help() {
    echo "usage: . ./scripts/cluster.sh exp_file exp_name [worker_n [device_n]] [--logdir dir] [extra_args]"
	echo "extra_args will be passed directly to experiment"
}


POSITIONAL=()

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --log_dir)
    log_dir="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

exp_file=$1

exp_name=$2

if [[ "$exp_file" = "" || "$exp_name" = "" ]]; then
    print_help
    return
fi

if [[ "$log_dir" = "" ]]; then
	log_dir=./log/$exp_name
fi

start_port=2242

worker_n=4
device_n=4

if [[ "$3" != "" ]]; then
    worker_n=$3
fi
if [[ "$4" != "" ]]; then
    device_n=$4
fi
extra_arg=""
if [[ "$5" != "" ]]; then
    extra_arg=$5
fi

cluster=$(python -c "import sys;_,port,worker=sys.argv;print \"{'ps':['localhost:\"+port+\"'], 'worker':[\" + \",\".join([\"'localhost:\"+str(i+int(port)+1)+\"'\" for i in range(int(worker))])+\"]}\"" $start_port $worker_n)
echo "cluster:      $cluster"
echo "device_n:     $device_n"
echo "extra_arg:    $extra_arg"
mkdir -p $log_dir
python $exp_file run --name $exp_name --cluster "$cluster" --job ps --logdir $log_dir > $log_dir/ps.txt 2>&1 &
end_i=$(expr $worker_n - 1)
echo "job $exp_name with worker: [ 0 .. $end_i ]"
for i in $(seq 0 $end_i)
do
	device=$(expr $i % $device_n)
    CUDA_VISIBLE_DEVICES=$device python $exp_file run --name $exp_name --cluster "$cluster" --job worker --index $i --logdir $log_dir $extra_arg > $log_dir/worker.$i.txt 2>&1 &
done

