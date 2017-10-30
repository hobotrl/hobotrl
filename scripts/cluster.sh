#!/usr/bin/env bash
#
#   usage: . ./scripts/cluster.sh exp_file exp_name [worker_n [device_n]]"
#

print_help() {
    echo "usage: . ./scripts/cluster.sh exp_file exp_name [worker_n [device_n]] [--logdir dir] [extra_args]"
	echo "extra_args will be passed directly to experiment"
}

log_dir=""
start_port=""
exp_file=""
exp_name=""
worker_n=""
device_n=""

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
    --start_port)
    start_port="$2"
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
shift

exp_name=$1
shift

if [[ "$exp_file" = "" || "$exp_name" = "" ]]; then
    print_help
    return
fi

if [[ "$log_dir" = "" ]]; then
	log_dir=./log/$exp_name
fi

if [[ "$start_port" = "" ]]; then
    start_port=2242
fi

worker_n=4
device_n=4

if [[ "$1" != "" ]]; then
    worker_n=$1
    shift
fi
if [[ "$1" != "" ]]; then
    device_n=$1
    shift
fi

extra_arg=$@

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
    if [[ "$device_n" = "0" ]]; then
        # without gpu
        device=""
    else
	    device=$(expr $i % $device_n)
    fi

    CUDA_VISIBLE_DEVICES=$device python $exp_file run --name $exp_name --cluster "$cluster" --job worker --index $i --logdir $log_dir $extra_arg > $log_dir/worker.$i.txt 2>&1 &
done

