#!/bin/bash
cd /root/ana
pkill -f dog_evolve || true
sleep 1
rm -f dog.log
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml \
  python3 -m agi.dog_evolve --gens 4 --pop 8 --ticks 80 --workers 8 --init walk_best.json \
  > dog.log 2>&1 < /dev/null &
disown
sleep 100
echo "=== dog.log ==="
head -12 dog.log 2>/dev/null
echo "进程: $(pgrep -fc dog_evolve)"
