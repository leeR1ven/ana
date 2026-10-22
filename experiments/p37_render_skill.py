"""按技能渲染帧（本地，手写 PNG），用来看清动作到底长什么样。

用法：python experiments/p37_render_skill.py walk0 5
"""
from __future__ import annotations

import json
import os
import struct
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BLUE = np.array([0.1, 0.2, 0.95], np.float32)

CFG = {  # skill: (初始高度, 姿态, 躯干力矩, 线索)
    "walk0": (0.24, "flat", 0.0, "side"),
    "biped": (0.26, "flat", -1.0, "side"),
    "reach": (0.26, "flat", -1.0, "high"),
    "getup2": (0.10, "side", 0.0, "side"),
    "recover": (0.12, "flip", 0.0, "side"),
    "orbit": (0.24, "flat", 0.0, "orbit"),
    "obstacle": (0.24, "flat", 0.0, "wall"),
}


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main(skill="walk0", n=5, ticks=150):
    import mujoco
    from agi.body_go2 import Go2Body
    from agi.dog_tasks import _ball, _ball_xy, _brain, _obs

    z, pose, torso, cue = CFG.get(skill, CFG["walk0"])
    kw = json.load(open(os.path.join(ROOT, "walk_seed3.json"), encoding="utf-8"))["kw"]
    sp, brain = _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.35)))
    body.reset()
    body.torso_cmd = torso
    if pose == "side":
        body.data.qpos[3:7] = np.array([0.7071, 0.7071, 0.0, 0.0])
    elif pose == "flip":
        body.data.qpos[3:7] = np.array([0.0, 1.0, 0.0, 0.0])
    body.data.qpos[2] = z
    body.data.qpos[7:19] = (np.array([0.0, 1.6, -2.6] * 4) if pose == "side"
                            else np.array([0.0, 1.0, -2.0] * 4))
    body.mj.mj_forward(body.model, body.data)

    renderer = mujoco.Renderer(body.model, height=360, width=480)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.azimuth, cam.elevation = 2.0, 135, -12
    out = os.path.join(ROOT, "frames")
    os.makedirs(out, exist_ok=True)
    px = sp.px
    step = max(1, ticks // n)
    k = 0
    for t in range(ticks):
        if cue == "high":
            img = _ball_xy(sp, 0.5, 0.15)
        elif cue == "orbit":
            img = _ball(sp, float(np.clip(0.5 + 0.45 * np.sin(2 * np.pi * t / 40.0), 0.05, 0.95)))
        elif cue == "wall":
            img = _ball(sp, 0.5).copy()
            img[px // 2:, px // 3: 2 * px // 3] = BLUE
        else:
            img = _ball(sp, 0.85)
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, img)))
        eng.step()
        body.step(brain.forces())
        if t % step == 0 and k < n:
            cam.lookat[:] = body.data.qpos[0:3]
            renderer.update_scene(body.data, cam)
            write_png(os.path.join(out, "%s_%d.png" % (skill, k)), renderer.render())
            k += 1
    print("ok", skill, k)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "walk0", int(a[1]) if len(a) > 1 else 5)
