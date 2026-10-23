"""渲染最远距离模型的帧（含平衡环驱动），确认是"走"还是"飞"。

用法：python experiments/p50_render_far.py 3
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(n=3, gap=40):
    import mujoco
    from agi.body_go2 import Go2Body
    from agi.dog_tasks import _ball, _brain, _drive_balance, _obs
    from experiments.p37_render_skill import write_png

    kw = json.load(open(os.path.join(ROOT, "walk_far.json"), encoding="utf-8"))["kw"]
    sp, brain = _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.35)))
    body.reset()
    body.data.qpos[2] = 0.27
    body.data.qpos[7:19] = np.array([0.0, 0.9, -1.8] * 4)
    body.mj.mj_forward(body.model, body.data)
    r = mujoco.Renderer(body.model, height=360, width=480)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.azimuth, cam.elevation = 1.8, 95, -6
    x0 = float(body.data.qpos[0])
    for i in range(n):
        for _ in range(gap):
            eng.clear_external()
            eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
            st = body.state()
            _drive_balance(eng, brain, float(st.get("roll", 0.0)), float(st.get("pitch", 0.0)),
                           int(kw.get("balance_sectors", 36)), int(kw.get("balance_per_sector", 20)))
            eng.step()
            body.step(brain.forces())
        cam.lookat[:] = body.data.qpos[0:3]
        r.update_scene(body.data, cam)
        write_png(os.path.join(ROOT, "frames", "far_%d.png" % i), r.render())
        print("帧%d: 位移 %.2f 米, 高度 %.3f" % (i, float(body.data.qpos[0]) - x0,
                                                 float(body.data.qpos[2])))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    raise SystemExit(main(int(a[0]) if a else 3))
