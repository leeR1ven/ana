#!/bin/bash
cd /root/ana || exit 1
export OMP_NUM_THREADS=1
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml OMP_NUM_THREADS=1 \
  python3 experiments/p49_walk_far.py --gens 60 --pop 40 --jobs 30 \
  > walkfar.log 2>&1 < /dev/null &
disown
sleep 150
echo PROCS=$(pgrep -fc p49_walk_far)
tail -5 walkfar.log
