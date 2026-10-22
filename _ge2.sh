#!/bin/bash
cd /root/ana || exit 1
export OMP_NUM_THREADS=1
pkill -f 'gait_evolve.py' 2>/dev/null || true
sleep 2
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml OMP_NUM_THREADS=1 \
  python3 experiments/p41_gait_evolve.py --gens 14 --pop 24 --jobs 22 --ticks 150 \
  > gait_evo.log 2>&1 < /dev/null &
disown
sleep 150
echo "=== 演化 ==="
tail -4 gait_evo.log 2>/dev/null
pgrep -fc p41_gait_evolve
