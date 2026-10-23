import json
import sys

import numpy as np

sys.path.insert(0, "/root/ana")
from agi.body_go2 import Go2Body
from agi.dog_tasks import _ball, _brain, _obs

kw = json.load(open("/root/ana/champion.json", encoding="utf-8"))["kw"]
sp, brain = _brain(kw)
eng = brain.engine
body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.35)))
body.reset()
body.data.qpos[2] = 0.27
body.data.qpos[7:19] = np.array([0.0, 0.9, -1.8] * 4)
body.mj.mj_forward(body.model, body.data)
x0 = float(body.data.qpos[0])
marks = []
for t in range(300):
    eng.clear_external()
    eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
    eng.step()
    body.step(brain.forces())
    if t % 50 == 49:
        marks.append((t + 1, round(float(body.data.qpos[0]) - x0, 3),
                      round(float(body.data.qpos[2]), 3)))
print("拍数 / 累计位移 / 高度：")
for m in marks:
    print("  %3d  %+.3f  高 %.3f" % m)
