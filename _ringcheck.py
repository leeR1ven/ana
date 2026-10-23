import json
import sys

import numpy as np

sys.path.insert(0, "/root/ana")
from agi.body_go2 import Go2Body
from agi.dog_tasks import _ball, _brain, _drive_balance, _obs

kw = json.load(open("/root/ana/walk_far.json", encoding="utf-8"))["kw"]
kw.update(balance_sectors=36, balance_per_sector=20, trot=True, balance_gain=1.5)
sp, brain = _brain(kw)
eng = brain.engine
ids = np.asarray(brain.motor["balance_ring"], dtype=np.int64).ravel()
print("平衡环神经元数:", ids.size, " 阈值样例:", np.round(np.asarray(eng.thr)[ids[:6]], 2))
body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.35)))
body.reset()
body.data.qpos[2] = 0.27
body.data.qpos[7:19] = np.array([0.0, 0.9, -1.8] * 4)
body.mj.mj_forward(body.model, body.data)
for t in range(45):
    eng.clear_external()
    eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
    st = body.state()
    _drive_balance(eng, brain, float(st.get("roll", 0.0)), float(st.get("pitch", 0.0)), 36, 20)
    eng.step()
    body.step(brain.forces())
    if t % 5 == 0:
        a = np.asarray(eng.act)[ids]
        e = np.asarray(eng.ext)[ids]
        print("拍%2d roll %+.2f | 环激活数 %2d (均值 %.2f) | ext 最大 %.2f"
              % (t, float(st.get("roll", 0)), int((a > 0.3).sum()), float(a.mean()), float(e.max())))
