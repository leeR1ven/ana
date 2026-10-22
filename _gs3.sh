#!/bin/bash
cd /root/ana || exit 1
pkill -f p39_gait_search || true
sleep 2
rm -f gait.log gait_11.log gait_12.log gait_13.log gait_14.log gait_best.json gait_top.json
# 关键：每个 worker 只用一个 BLAS 线程，避免几十个进程互相抢
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
for s in 21 22; do
  setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml OMP_NUM_THREADS=1 \
    python3 experiments/p39_gait_search.py --n 1500 --jobs 24 --ticks 150 --seed $s \
    > gait_$s.log 2>&1 < /dev/null &
  disown
done
sleep 90
echo "=== 进程 ==="
pgrep -fc p39_gait_search
for s in 21 22; do echo "--- $s ---"; tail -2 gait_$s.log 2>/dev/null; done
