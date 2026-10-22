# -*- coding: utf-8 -*-
"""节律前进实验：给 12 条肌肉一个交替节律（对角步），看机身能不能往前走。

这一步**只用 CPG**（不用脑），目的有两个：
  1. 确认"行走"这件事在 Go2 仿真里是可判分的（位移 > 阈值、不摔）；
  2. 为"把节律接到运动区"提供基线（脑接上后应该不比纯 CPG 差太多）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build_spec  # noqa: E402
from agi.body_go2 import Go2Body  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=200)
    p.add_argument("--period", type=int, default=40)
    p.add_argument("--amp", type=float, default=0.35)
    p.add_argument("--base", type=float, default=0.5)
    args = p.parse_args(argv)

    sp = build_spec("full")
    body = Go2Body(posture_offset=0.12)
    body.reset()
    body.data.qpos[2] = 0.20
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])

    # 对角步：腿 0、2 同相，腿 1、3 反相
    phase = np.array([0.0, np.pi, 0.0, np.pi])
    heights, xs = [], []
    for t in range(args.ticks):
        th = 2 * np.pi * t / args.period
        forces = np.zeros(12, np.float32)
        for leg in range(4):
            forces[leg * 3:(leg + 1) * 3] = args.base + args.amp * np.sin(th + phase[leg])
        forces = np.clip(forces, 0, 1)
        body.step(forces)
        s = body.state()
        heights.append(s["height"])
        xs.append(float(body.data.qpos[0]))
        if (t + 1) % max(1, args.ticks // 5) == 0:
            print(f"  t={t+1:3d} 高度 {s['height']:.3f} 位移 {xs[-1]-x0:+.3f} 摔倒 {body.fallen()}")
    dx = xs[-1] - x0
    ok = (not body.fallen()) and abs(dx) > 0.3
    print(f"位移 {dx:+.3f} 末段高度 {np.mean(heights[-20:]):.3f} 摔倒 {body.fallen()}")
    print(f"能走 {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
