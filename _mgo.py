import json
import sys

import numpy as np

sys.path.insert(0, "/root/ana")
from agi.world import _blob
from agi.dog_tasks import _brain, _obs
from agi.body_go2 import Go2Body

kw = json.load(open("/root/ana/walk_seed3.json", encoding="utf-8"))["kw"]


def run(size, ticks=120):
    sp, brain = _brain(kw)
    eng = brain.engine
    px = sp.px
    img = np.maximum(np.full((px, px, 3), 0.05, np.float32),
                     _blob(px, 0.85, 0.5, size, (0.95, 0.2, 0.1)))
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.35)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, img)))
        eng.step()
        body.step(brain.forces())
    return abs(float(body.data.qpos[0]) - x0)


for size in (0.10, 0.18, 0.26):
    print("球大小 %.2f (越大越近): 走到 %.3f 米" % (size, run(size)))
