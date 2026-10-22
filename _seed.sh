#!/bin/bash
cd /root/ana
pkill -f dog_evolve || true
sleep 1
rm -f walk_best.json walk2.log
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml \
  python3 experiments/p29_walk_par.py --n 400 --jobs 55 --ticks 100 > walk2.log 2>&1 < /dev/null &
disown
sleep 60
tail -3 walk2.log 2>/dev/null
echo "进程: $(pgrep -fc p29_walk_par)"
ls -l walk_best.json 2>/dev/null | awk '{print "best 文件:", $5, "字节"}'
