#!/bin/bash
cd /root/ana
tail -2 walk2.log
echo "扫描进程: $(pgrep -fc p29_walk_par)"
python3 - <<'PY'
import json, os
if os.path.exists('walk_best.json'):
    d = json.load(open('walk_best.json'))
    print('best dx = %.4f' % d['dx'])
    print('kw =', d['kw'])
else:
    print('还没写出 best')
PY
