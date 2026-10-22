#!/bin/bash
cd /root/ana || exit 1
pkill -f dog_evolve || true
sleep 1
rm -f dog.log dog_evolve_result.json
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml \
  python3 -m agi.dog_evolve --gens 12 --pop 16 --ticks 100 --workers 14 \
  --init walk_seed.json > dog.log 2>&1 < /dev/null &
disown
sleep 100
grep '代：技能' dog.log | tail -3
