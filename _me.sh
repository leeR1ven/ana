#!/bin/bash
cd /root/ana || exit 1
export OMP_NUM_THREADS=1
pkill -f 'gait_evolve.py' 2>/dev/null || true
pkill -f 'p42_multi_evolve' 2>/dev/null || true
sleep 2
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml OMP_NUM_THREADS=1 \
  python3 experiments/p42_multi_evolve.py --gens 14 --pop 24 --jobs 26 --ticks 120 \
  > multi.log 2>&1 < /dev/null &
disown
sleep 240
echo "=== 多技能演化（含融合）==="
grep '代' multi.log | tail -5
tail -3 multi.log
