#!/bin/bash


_term(){
	echo "Caught SIGTERM signal!"
	sleep 5
	echo "After sleep"
	kill -TERM "$child" 2>/dev/null
}

trap _term SIGTERM


INSTALL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../../../../"

ulimit -s 40000

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INSTALL_DIR/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$INSTALL_DIR:$INSTALL_DIR/lib:$INSTALL_DIR/python

arguments=""
[[ $# -gt 0 ]] && arguments+="--vehicle_configuration $1"
[[ $# -gt 1 ]] && arguments+=" --map_file $2"

cd ${INSTALL_DIR}/python
python decision_making/src/dm_main.py ${arguments} --pid $$ &

child=$!
wait "$child"