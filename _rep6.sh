#!/bin/bash
cd /root/ana || exit 1
echo "=== 代际 ==="
grep '代：技能' dog.log | tail -6
echo "=== 进程 ==="
pgrep -fc dog_evolve || echo 0
echo "=== 结果 ==="
if [ -f dog_evolve_result.json ]; then
python3 - <<'PY'
import json
d = json.load(open('dog_evolve_result.json'))
print('skills:', d.get('skills'))
print('best_by_skill:', d.get('best_by_skill'))
kw = d.get('best_kw') or {}
print('clock_motor:', kw.get('clock_motor'), 'posture_offset:', kw.get('posture_offset'))
PY
else
echo "结果未写（仍在跑）"
fi
