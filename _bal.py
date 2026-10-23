import json
import sys

import numpy as np

sys.path.insert(0, "/root/ana")
from agi.body_go2 import Go2Body
from agi.dog_tasks import _ball, _brain, _drive_balance, _obs

base = json.load(open("/root/ana/champion.json", encoding="utf-8"))["kw"]
for sec, per, gain in ((0, 0, 0.0), (36, 20, 0.5), (36, 20, 2.0)):
    kw = dict(base)
    kw.update(balance_sectors=sec, balance_per_sector=per, balance_gain=gain)
    sp, brain = _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.35)))
    body.reset()
    body.data.qpos[2] = 0.27
    body.data.qpos[7:19] = np.array([0.0, 0.9, -1.8] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    fell_at = None
    for t in range(300):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        if sec:
            st = body.state()
            _drive_balance(eng, brain, float(st.get("roll", 0.0)),
                           float(st.get("pitch", 0.0)), sec, per)
        eng.step()
        body.step(brain.forces())
        if fell_at is None and body.fallen():
            fell_at = t
    dx = float(body.data.qpos[0]) - x0
    print("平衡环 %2d×%2d 增益 %.1f: 位移 %+.2f 米  摔倒于 %s 拍"
          % (sec, per, gain, dx, fell_at if fell_at is not None else "没倒"))
