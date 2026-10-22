#!/bin/bash
cd /root/ana
echo "=== dog.log ==="
tail -20 dog.log
echo "=== 进程 ==="
pgrep -fc dog_evolve
echo "=== result json ==="
python3 - <<'PY'
import json, os
p = 'dog_evolve_result.json'
if os.path.exists(p):
    d = json.load(open(p))
    print('技能清单:', d.get('skills'))
    bbs = d.get('best_by_skill', {})
    for k, v in bbs.items():
        print('  %-14s best metric %.3f' % (k, v.get('metric', 0)))
else:
    print('结果文件还没写')
PY
