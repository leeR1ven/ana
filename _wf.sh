#!/bin/bash
cd /root/ana || exit 1
export OMP_NUM_THREADS=1
pkill -f 'night.sh' 2>/dev/null || true
pkill -f 'multi2.py' 2>/dev/null || true
sleep 2
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml OMP_NUM_THREADS=1 \
  python3 experiments/p49_walk_far.py --gens 24 --pop 30 --jobs 28 --ticks 300 \
  > walkfar.log 2>&1 < /dev/null &
disown
sleep 180
echo "=== 最远距离演化 ==="
tail -6 walkfar.log 2>/dev/null
