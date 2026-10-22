#!/bin/bash
cd /root/ana || exit 1
pkill -f p39_gait_search || true
sleep 1
rm -f gait.log gait_best.json
for s in 11 12 13 14; do
  setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml \
    python3 experiments/p39_gait_search.py --n 1000 --jobs 14 --ticks 150 --seed $s \
    > gait_$s.log 2>&1 < /dev/null &
  disown
done
sleep 60
echo "=== 启动状态 ==="
pgrep -fc p39_gait_search
for s in 11 12 13 14; do echo "--- seed $s ---"; tail -2 gait_$s.log 2>/dev/null; done
