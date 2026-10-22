#!/bin/bash
cd /root/ana || exit 1
export OMP_NUM_THREADS=1
pkill -f 'gait_evolve.py' 2>/dev/null || true
sleep 2
# 用目前最好的两只当种子之一（直立但会栽的 gait_seed2），加姿态高度评分后重跑
cp -f gait_seed2.json gait_top1.json
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml OMP_NUM_THREADS=1 \
  python3 experiments/p41_gait_evolve.py --gens 16 --pop 26 --jobs 26 --ticks 150 \
  > gait_evo.log 2>&1 < /dev/null &
disown
sleep 180
echo "=== 演化（含姿态高度评分）==="
grep '代' gait_evo.log | tail -4
