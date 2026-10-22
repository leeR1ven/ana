#!/bin/bash
cd /root/ana || exit 1
pkill -f p39_gait_search || true
pkill -f dog_evolve || true
sleep 1
rm -f gait.log gait_best.json
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml \
  python3 experiments/p39_gait_search.py --n 400 --jobs 16 --ticks 150 \
  > gait.log 2>&1 < /dev/null &
disown
sleep 240
echo "=== 进度 ==="
grep -c '#[0-9]' gait.log || true
tail -6 gait.log
