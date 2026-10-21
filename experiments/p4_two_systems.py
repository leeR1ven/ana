# -*- coding: utf-8 -*-
"""P4：两套控制 + 前额叶压制。

架子里有两套通往肌肉的路径：
  1. 前额叶 → 运动区（我们有意识的控制）；
  2. 各区自己 → 运动区（反射/习惯，出生接线、天生难改）。
前额叶的正常状态会通过抑制细胞把第二套压住——"整体上前额叶赢"。

实验：把狗摆成不同倾角，分别在
  A) 前额叶运动组被关掉（只剩反射）
  B) 前额叶正常（应该压制反射）
下，量反射中继的活跃程度和最终力度。

    python experiments/p4_two_systems.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec                                        # noqa: E402
from agi.body_go2 import Go2Body                                        # noqa: E402
from agi.loop import Cortex                                             # noqa: E402


def tilt_roll(body, roll: float, height: float = 0.27) -> None:
    """把机身摆成指定的 roll（不动关节）。"""
    q = np.zeros(4)
    q[0] = np.cos(roll / 2)      # w
    q[1] = np.sin(roll / 2)      # x
    body.data.qpos[2] = height
    body.data.qpos[3:7] = q
    body.data.qvel[:] = 0.0
    body.mj.mj_forward(body.model, body.data)


def measure(brain, body, ctx, roll: float, pfc_off: bool, ticks: int = 40):
    eng = brain.engine
    mpfc = brain.motor["mpfc"]
    old = eng.thr[mpfc].copy()
    if pfc_off:
        eng.set_thr(mpfc, thr=1e9, soft=1.0)
    # 冻结物理：只把"倾斜姿态"喂给大脑，看它想怎么动（不跑动力学）
    body.reset()
    tilt_roll(body, roll)
    obs = body.observe(brain.spec)
    eng.act[:] = 0.0
    eng.reset_adaptation()
    reflex_n, force = [], []
    for _ in range(ticks):
        tk = ctx.tick(obs, record=False)
        reflex_n.append(int((eng.act_np[brain.motor["reflex"]] > 0.35).sum()))
        force.append(tk.forces.copy())
    eng.set_thr(mpfc, thr=old, soft=eng.soft[mpfc].copy())
    return float(np.mean(reflex_n[-10:])), np.mean(np.stack(force[-10:]), axis=0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="full")
    p.add_argument("--ticks", type=int, default=40)
    args = p.parse_args(argv)

    sp = build_spec(args.preset, pfc_topk=0.005, pfc_adapt=0.01)
    brain = build(sp, verbose=False)
    body = Go2Body()
    ctx = Cortex(brain, trace=False)
    print("倾角 → 反射中继活跃 / 力度均值（A=只留反射，B=前额叶正常）")
    for roll in (-0.4, -0.2, 0.0, 0.2, 0.4):
        na, fa = measure(brain, body, ctx, roll, pfc_off=True, ticks=args.ticks)
        nb, fb = measure(brain, body, ctx, roll, pfc_off=False, ticks=args.ticks)
        print(f"  roll {roll:+.1f}:  A 反射 {na:3.0f} 力度 {fa.mean():.2f}   |   "
              f"B 反射 {nb:3.0f} 力度 {fb.mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
