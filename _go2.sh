#!/bin/bash
set -e
cd /root
rm -rf go2models && mkdir -p go2models
python3 - <<'PY'
import zipfile
zipfile.ZipFile('/root/go2.zip').extractall('/root/go2models')
print('models extracted')
PY
ls /root/go2models | head
cd /root/ana
# 用刚上传的 body_go2.py（支持 ANA_GO2_XML）
export ANA_GO2_XML=/root/go2models/scene.xml
python3 -c "import sys,os; sys.path.insert(0,'.'); from agi.body_go2 import Go2Body; b=Go2Body(posture_offset=0.12); b.reset(); print('body ok', round(b.state()['height'],3))"
rm -f walk.log walk_best.json
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml python3 experiments/p29_walk_par.py --n 3000 --jobs 60 --ticks 100 > walk.log 2>&1 < /dev/null &
disown
sleep 10
head -3 walk.log 2>/dev/null
pgrep -fc p29_walk_par
