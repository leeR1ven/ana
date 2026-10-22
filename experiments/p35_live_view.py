"""在本地打开 MuJoCo 实时窗口，看机器狗跑某个技能。

用法（系统 Python，需桌面环境）：
    python experiments/p35_live_view.py walk0 20
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(skill: str = "walk0", seconds: float = 20.0) -> int:
    import mujoco.viewer
    from agi.body_go2 import Go2Body
    from agi.dog_tasks import _ball, _brain, _obs

    kw = json.load(open(os.path.join(ROOT, "walk_seed3.json"), encoding="utf-8"))["kw"]
    sp, brain = _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.35)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)

    with mujoco.viewer.launch_passive(body.model, body.data) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        t0 = time.time()
        while viewer.is_running() and time.time() - t0 < seconds:
            eng.clear_external()
            eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
            eng.step()
            body.step(brain.forces())
            viewer.cam.lookat[:] = body.data.qpos[0:3]
            viewer.sync()
            time.sleep(0.02)
    print("窗口已关闭")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args[0] if args else "walk0",
                          float(args[1]) if len(args) > 1 else 20.0))
