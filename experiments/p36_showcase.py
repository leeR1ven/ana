"""在本地实时窗口里依次演示多个动作（巡演）。

用法（系统 Python，需桌面环境）：
    python experiments/p36_showcase.py
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

    kw = json.load(open(os.path.join(ROOT, "walk_seed3.json"), encoding="utf-8"))["kw"]
    off = float(kw.get("posture_offset", 0.35))

    # (名字, 初始高度, 是否侧躺, 是否翻转, 躯干指令, 线索类型, 秒数)
    plan = [
        ("走路 walk0", 0.24, False, False, 0.0, "side", 6),
        ("侧躺起身 getup2", 0.10, True, False, 0.0, "side", 6),
        ("四脚朝天翻回 recover", 0.12, False, True, 0.0, "side", 7),
        ("两腿站立 biped", 0.26, False, False, -1.0, "side", 6),
        ("够高处 reach", 0.26, False, False, -1.0, "high", 6),
        ("追绕圈的球 orbit", 0.24, False, False, 0.0, "orbit", 7),
        ("看到墙停下 obstacle", 0.24, False, False, 0.0, "wall", 6),
        ("转向 turn1", 0.24, False, False, 0.0, "side", 5),
    ]

    for name, z, side, flipped, torso, cue, secs in plan:
        sp, brain = _brain(kw)
        eng = brain.engine
        body = Go2Body(posture_offset=off)
        body.reset()
        body.torso_cmd = torso
        if flipped:
            body.data.qpos[3:7] = np.array([0.0, 1.0, 0.0, 0.0])
        elif side:
            body.data.qpos[3:7] = np.array([0.7071, 0.7071, 0.0, 0.0])
        body.data.qpos[2] = z
        body.data.qpos[7:19] = np.array([0.0, 1.6, -2.6] * 4) if side else np.array([0.0, 1.0, -2.0] * 4)
        body.mj.mj_forward(body.model, body.data)

        with mujoco.viewer.launch_passive(body.model, body.data) as viewer:
            viewer.cam.distance = 2.2
            viewer.cam.elevation = -18
            viewer.cam.azimuth = 135
            t0 = time.time()
            t = 0
            while viewer.is_running() and time.time() - t0 < secs:
                if cue == "high":
                    img = _ball_xy(sp, 0.5, 0.15)
                elif cue == "orbit":
                    x = 0.5 + 0.45 * np.sin(2 * np.pi * t / 40.0)
                    img = _ball(sp, float(np.clip(x, 0.05, 0.95)))
                elif cue == "wall":
                    img = _ball(sp, 0.5).copy()
                    px = sp.px
                    img[px // 2:, px // 3: 2 * px // 3] = BLUE
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
    print("巡演结束")
    return 0


def _dummy():
    raise RuntimeError("unused")


if __name__ == "__main__":
    raise SystemExit(main())
