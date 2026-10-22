#!/bin/bash
cd /root/ana
pkill -f p29_walk_par || true
sleep 2
rm -f walk_best.json walk.log
setsid nohup python3 experiments/p29_walk_par.py --n 3000 --jobs 60 --ticks 100 > walk.log 2>&1 < /dev/null &
disown
sleep 8
echo "--- 启动后 ---"
head -3 walk.log 2>/dev/null
pgrep -fc p29_walk_par
