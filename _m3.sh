#!/bin/bash
cd /root/ana || exit 1
export OMP_NUM_THREADS=1
pkill -f 'p43_multi2' 2>/dev/null || true
pkill -f 'p42_multi_evolve' 2>/dev/null || true
sleep 2
# 用当前最好的模型当种子，融合孩子数量加大
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml OMP_NUM_THREADS=1 \
  python3 experiments/p43_multi2.py --gens 30 --pop 24 --jobs 26 --ticks 120 \
  > multi3.log 2>&1 < /dev/null &
disown
sleep 240
echo "=== 平均融合 + 大批后代 ==="
grep '代' multi3.log | tail -8
