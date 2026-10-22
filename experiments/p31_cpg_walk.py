# -*- coding: utf-8 -*-
"""验证"缺节律"：按关节给固定节律（对角步），看机身能不能前进。

肌肉编号：12 条 = 4 腿 × 3 关节。这里假设顺序是 [hip, thigh, calf] × 4。

用法（系统 Python，需 mujoco）：
    python experiments/p31_cpg_walk.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi.body_go2 import Go2Body   # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=160)
    p.add_argument("--period", type=int, default=30)
    p.add_argument("--amp_thigh", type=float, default=0.45)
    p.add_argument("--amp_calf", type=float, default=0.45)
    p.add_argument("--base", type=float, default=0.35)
    a = p.parse_args(argv)

    body = Go2Body(posture_offset=0.12)
    body.reset()
    body.data.qpos[2] = 0.26
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0, y0 = float(body.data.qpos[0]), float(body.data.qpos[1])

    # 对角步：左前/右后 同相，右前/左后 反相
    phase = np.array([0.0, np.pi, np.pi, 0.0])
    xs, hs = [], []
    for t in range(a.ticks):
        th = 2 * np.pi * t / a.period
        f = np.zeros(12, np.float32)
        for leg in range(4):
            ph = th + phase[leg]
            f[leg * 3 + 0] = a.base * 0.5
            f[leg * 3 + 1] = a.base + a.amp_thigh * np.sin(ph)
            f[leg * 3 + 2] = a.base + a.amp_calf * max(0.0, np.sin(ph + np.pi / 2))
        body.step(np.clip(f, 0, 1))
        xs.append(float(body.data.qpos[0]))
        hs.append(body.state()["height"])
    dx, dy = xs[-1] - x0, float(body.data.qpos[1]) - y0
    print(f"位移 dx {dx:+.3f} dy {dy:+.3f} 末段高度 {np.mean(hs[-20:]):.3f} 摔倒 {body.fallen()}")
    print("会走" if (not body.fallen() and abs(dx) > 0.10) else "还不会走")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
