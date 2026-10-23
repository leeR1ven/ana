import json
import sys

import numpy as np

sys.path.insert(0, "/root/ana")
from agi.body_go2 import Go2Body
from agi.dog_tasks import _ball, _brain, _drive_balance, _obs

kw = json.load(open("/root/ana/walk_far.json", encoding="utf-8"))["kw"]
kw.update(balance_sectors=36, balance_per_sector=20, trot=True)
sp, brain = _brain(kw)
eng = brain.engine
body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.35)))
body.reset()
body.data.qpos[2] = 0.27
body.data.qpos[7:19] = np.array([0.0, 0.9, -1.8] * 4)
body.mj.mj_forward(body.model, body.data)
print("拍  高   roll   pitch   x")
for t in range(70):
    eng.clear_external()
    eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
    st = body.state()
    _drive_balance(eng, brain, float(st.get("roll", 0.0)), float(st.get("pitch", 0.0)), 36, 20)
    eng.step()
    body.step(brain.forces())
    if t % 6 == 0 or body.fallen():
        st = body.state()
        print("%3d %.3f %+.2f %+.2f %+.3f" % (t, float(body.data.qpos[2]),
              float(st.get("roll", 0)), float(st.get("pitch", 0)),
              float(body.data.qpos[0])))
    if body.fallen():
        print("倒了：roll=%.2f pitch=%.2f 高=%.3f" % (float(st.get("roll", 0)),
              float(st.get("pitch", 0)), float(body.data.qpos[2])))
        break
