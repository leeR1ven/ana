#!/bin/bash
cd /root/ana
pkill -f p29_walk_par || true
sleep 2
echo "扫描已停，最好位移: $(python3 -c "import json;print(json.load(open('walk_best.json'))['dx'])" 2>/dev/null)"
rm -f dog.log
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml \
  python3 -m agi.dog_evolve --gens 3 --pop 6 --ticks 80 --init walk_best.json \
  > dog.log 2>&1 < /dev/null &
disown
sleep 20
echo "--- dog.log 开头 ---"
head -6 dog.log 2>/dev/null
echo "dog 进程: $(pgrep -fc dog_evolve)"
