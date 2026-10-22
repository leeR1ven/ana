# -*- coding: utf-8 -*-
"""测"动作幅度"是否是走路的关键：改 posture_offset（关节目标偏移范围）。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agi.body_go2 import Go2Body   # noqa: E402


def run(offset, ticks=160, period=30):
    body = Go2Body(posture_offset=offset)
    body.reset()
    body.data.qpos[2] = 0.26
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    phase = np.array([0.0, np.pi, np.pi, 0.0])
    xs, hs = [], []
    for t in range(ticks):
        th = 2 * np.pi * t / period
        f = np.zeros(12, np.float32)
        for leg in range(4):
            ph = th + phase[leg]
            f[leg * 3 + 0] = 0.5
            f[leg * 3 + 1] = 0.5 + 0.45 * np.sin(ph)
            f[leg * 3 + 2] = 0.5 + 0.45 * max(0.0, np.sin(ph + np.pi / 2))
        body.step(np.clip(f, 0, 1))
        xs.append(float(body.data.qpos[0]))
        hs.append(body.state()["height"])
    return float(xs[-1] - x0), float(np.mean(hs[-20:])), bool(body.fallen())


for off in (0.12, 0.4, 0.8, 1.2):
    dx, h, f = run(off)
    print("offset %.2f: dx %+.3f 末高 %.3f 摔倒 %s" % (off, dx, h, f))
