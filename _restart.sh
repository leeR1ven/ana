#!/bin/bash
cd /root/ana || exit 1
pkill -f 'night.sh' 2>/dev/null || true
pkill -f 'p43_multi2' 2>/dev/null || true
sleep 3
echo "[$(date '+%H:%M:%S')] ===== 重启（带防停滞换种子）=====" >> night.log
setsid nohup bash /root/night.sh >> /dev/null 2>&1 < /dev/null &
disown
sleep 40
pgrep -fc python3
tail -3 night.log
