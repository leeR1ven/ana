#!/bin/bash
cd /root/ana || exit 1
pkill -f dog_evolve || true
sleep 1
rm -f dog.log dog_evolve_result.json
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml ANA_BIDIR=1 \
  python3 -m agi.dog_evolve --gens 20 --pop 32 --ticks 100 --workers 14 \
  --init walk_seed3.json > dog.log 2>&1 < /dev/null &
disown
sleep 240
echo "=== 进度 ==="
grep '代：技能' dog.log | tail -3
echo "=== gait 情况 ==="
grep -c 'gait' dog.log || true
