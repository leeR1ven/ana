# -*- coding: utf-8 -*-
"""两阶段起身实验（先放松、再站起），只用身体、不用脑。

目的：确认"侧躺 → 放松躺平 → 站起"这条两阶段路径在仿真里可行。
如果可行，再谈用时间环调度两段命令；如果不可行，说明需要更细的动作接口。
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
    p.add_argument("--relax", type=int, default=60)
    p.add_argument("--stand", type=int, default=120)
    args = p.parse_args(argv)

    sp = build_spec("full")
    body = Go2Body(posture_offset=0.12)
    body.reset()
    body.data.qpos[2] = 0.10
    body.data.qpos[3:7] = np.array([0.7071, 0.7071, 0.0, 0.0])   # 侧躺
    body.data.qpos[7:19] = np.array([0.0, 1.6, -2.6] * 4)
    body.mj.mj_forward(body.model, body.data)
    print(f"起始高度 {body.state()['height']:.3f} 摔倒 {body.fallen()}")

    heights = []
    for t in range(args.relax + args.stand):
        if t < args.relax:
            forces = np.zeros(12, np.float32)          # 第一阶段：放松
        else:
            forces = np.full(12, 0.5, np.float32)      # 第二阶段：站起来
        body.step(forces)
        s = body.state()
        heights.append(s["height"])
        if (t + 1) % 45 == 0:
            print(f"  t={t+1:3d} 高度 {s['height']:.3f} 倾角 {s['roll']:+.2f}/{s['pitch']:+.2f} 摔倒 {body.fallen()}")

    ok = (not body.fallen()) and float(np.mean(heights[-20:])) > 0.20
    print(f"末段高度 {np.mean(heights[-20:]):.3f} 摔倒 {body.fallen()}")
    print(f"两阶段起身成功 {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
