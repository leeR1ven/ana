#!/bin/bash
cd /root/ana || exit 1
export OMP_NUM_THREADS=1
pkill -f 'p42_multi_evolve' 2>/dev/null || true
pkill -f 'p43_multi2' 2>/dev/null || true
sleep 2
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml OMP_NUM_THREADS=1 \
  python3 experiments/p43_multi2.py --gens 24 --pop 22 --jobs 24 --ticks 120 \
  > multi2.log 2>&1 < /dev/null &
disown
sleep 240
echo "=== 演化 ==="
grep '代' multi2.log | tail -6
