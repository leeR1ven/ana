#!/bin/bash
# 通宵连续演化：一轮跑完自动开下一轮，日志按轮次分开，模型带版本号
cd /root/ana || exit 1
# 先停掉旧的演化进程（写在这里，避免外层命令行匹配到自己）
pkill -f 'multi_evolve.py' 2>/dev/null || true
pkill -f 'multi2.py' 2>/dev/null || true
sleep 2
export OMP_NUM_THREADS=1
export ANA_GO2_XML=/root/go2models/scene.xml
i=0
while true; do
  i=$((i+1))
  # 防停滞：每 3 轮换一个种子（从历史代存档里随机挑），避免一直围着同一只转
  if [ $((i % 3)) -eq 0 ]; then
    pick=$(ls -t multi2_g*.json multi2_best.json 2>/dev/null | shuf -n 1)
    if [ -n "$pick" ]; then
      cp -f "$pick" gait_top1.json
      echo "[$(date '+%H:%M:%S')] 换种子 <- $pick" >> night.log
    fi
  fi
  echo "[$(date '+%H:%M:%S')] ===== 第 $i 轮开始 =====" >> night.log
  python3 experiments/p43_multi2.py --gens 20 --pop 24 --jobs 26 --ticks 120 \
      >> night_$i.log 2>&1
  echo "[$(date '+%H:%M:%S')] ===== 第 $i 轮结束 =====" >> night.log
  # 每轮结束做一次全代体检，把最全面的存成 champion
  python3 experiments/p46_audit_gens.py >> night_audit.log 2>&1
  sleep 5
done
