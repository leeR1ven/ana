#!/bin/bash
# 看门狗：p49 停了就自动重启，保证长时间连续训练
cd /root/ana || exit 1
while true; do
  n=$(pgrep -fc p49_walk_far)
  if [ "$n" -lt 2 ]; then
    echo "[$(date '+%H:%M:%S')] 检测到停止($n)，重启" >> watchdog.log
    bash /root/wf5.sh >> /dev/null 2>&1
  fi
  sleep 300
done
