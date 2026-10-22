"""渲染机器狗动作的关键帧为 PNG（不依赖 PIL/imageio，手写 PNG 编码）。

用法（系统 Python，需 mujoco + 桌面环境）：
    python experiments/p34_render_frames.py walk
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


def write_png(path: str, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def main(skill: str = "walk0", n: int = 6, ticks: int = 180) -> int:
    import mujoco
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

    renderer = mujoco.Renderer(body.model, height=360, width=480)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.azimuth, cam.elevation = 2.0, 135, -18

    out = os.path.join(ROOT, "frames")
    os.makedirs(out, exist_ok=True)
    step = max(1, ticks // n)
    k = 0
    for t in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        body.step(brain.forces())
        if t % step == 0 and k < n:
            cam.lookat[:] = body.data.qpos[0:3]
            renderer.update_scene(body.data, cam)
            write_png(os.path.join(out, "%s_%02d.png" % (skill, k)), renderer.render())
            k += 1
    print("wrote", k, "frames to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(sys.argv[1:] or ["walk0"])))
