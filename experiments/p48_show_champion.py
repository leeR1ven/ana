"""用冠军模型在本地窗口里依次演示几个动作（最接近"看视频"）。

用法：python experiments/p48_show_champion.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BLUE = np.array([0.1, 0.2, 0.95], np.float32)


def main() -> int:
    import mujoco.viewer
    from agi.body_go2 import Go2Body
    from agi.dog_tasks import _ball, _ball_xy, _brain, _obs

    kw = json.load(open(os.path.join(ROOT, "champion.json"), encoding="utf-8"))["kw"]
    off = float(kw.get("posture_offset", 0.35))
    plan = [
        ("跑（gait）", "side", 7),
        ("追绕圈的球（orbit）", "orbit", 7),
        ("转向（turn1）", "turn1", 5),
        ("够高处（reach）", "high", 6),
    ]
    for name, cue, secs in plan:
        sp, brain = _brain(kw)
        eng = brain.engine
        body = Go2Body(posture_offset=off)
        body.reset()
        body.data.qpos[2] = 0.27
        body.data.qpos[7:19] = np.array([0.0, 0.9, -1.8] * 4)
        body.mj.mj_forward(body.model, body.data)
        with mujoco.viewer.launch_passive(body.model, body.data) as viewer:
            viewer.cam.distance = 1.7
            viewer.cam.elevation = -8
            viewer.cam.azimuth = 95
            t0, t = time.time(), 0
            while viewer.is_running() and time.time() - t0 < secs:
                if cue == "orbit":
                    x = 0.5 + 0.45 * np.sin(2 * np.pi * t / 40.0)
                    img = _ball(sp, float(np.clip(x, 0.05, 0.95)))
                elif cue == "high":
                    img = _ball_xy(sp, 0.5, 0.15)
                elif cue == "turn1":
                    img = _ball(sp, 0.08)
                else:
                    img = _ball(sp, 0.85)
                eng.clear_external()
                eng.set_input(brain.encoder.encode(_obs(sp, img)))
                eng.step()
                body.step(brain.forces())
                viewer.cam.lookat[:] = body.data.qpos[0:3]
                viewer.sync()
                t += 1
                time.sleep(0.02)
        print("演示完：", name, flush=True)
    print("全部结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
