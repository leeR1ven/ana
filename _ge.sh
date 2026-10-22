#!/bin/bash
cd /root/ana || exit 1
export OMP_NUM_THREADS=1
# 搜索还在跑就让它跑（它不占满核心），这里并行开演化
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml OMP_NUM_THREADS=1 \
  python3 experiments/p41_gait_evolve.py --gens 12 --pop 20 --jobs 20 --ticks 150 \
  > gait_evo.log 2>&1 < /dev/null &
disown
sleep 200
echo "=== 演化进度 ==="
tail -6 gait_evo.log 2>/dev/null
