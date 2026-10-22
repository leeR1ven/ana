# -*- coding: utf-8 -*-
"""P5：用极少的连接验证架构——一条"站起来"的命令链。

不手工连成千上万条边，只加 **1 个命令神经元 + 60 条边**：
命令 → 每块肌肉的前 5 档力度细胞（权重 6），于是 12 块肌肉各出 0.5 档力，
狗从趴着被撑起来站住。目的不是"会站"，是验证
   大脑（神经元）→ 运动区（力度档）→ 身体（PD）→ 站在地上
这条链是通的。

    python experiments/p5_stand.py --ticks 120
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


def install_stand(engine, brain) -> np.ndarray:
    """加 1 个命令神经元，接到每块肌肉前 5 档力度细胞上。"""
    cmd = engine.grow_units(1, kind="motor_relay", region="motor", depth=6,
                            thr=0.0, soft=1.0, pos=np.array([[0.84, 0.92, 0.5]],
                                                            np.float32))
    src, dst, w = [], [], []
    for m in range(brain.spec.n_muscle):
        f = brain.motor["force"][m]
        src.append(np.full(5, cmd[0], np.int32))
        dst.append(f[:5])
        w.append(np.full(5, 6.0, np.float32))
    engine.add_edges(np.concatenate(src), np.concatenate(dst),
                     np.concatenate(w), group=60)
    engine.rebuild_backend()
    return cmd


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--preset", default="full")
    p.add_argument("--offset", type=float, default=0.12)
    p.add_argument("--k_near", type=int, default=0)      # 0 = 用预设值
    p.add_argument("--region_links", type=int, default=-1)
    p.add_argument("--n_time", type=int, default=0)
    args = p.parse_args(argv)

    sp = build_spec(args.preset)
    if args.k_near:
        sp.k_near = args.k_near
    if args.region_links >= 0:
        sp.region_links = args.region_links
    if args.n_time:
        sp.n_time = args.n_time
        sp.hippo_bind = sp.hippo_bind and args.n_time > 0
    brain = build(sp, verbose=False)
    eng = brain.engine
    cmd = install_stand(eng, brain)
    body = Go2Body(posture_offset=args.offset)
    body.reset()
    # 先趴下：把机身压低、关节收到趴姿，模拟"还没站起来"
    body.data.qpos[2] = 0.10
    body.data.qpos[7:19] = np.array([0.0, 1.6, -2.6] * 4)
    body.mj.mj_forward(body.model, body.data)
    print(f"起点高度 {body.state()['height']:.3f}  （趴着）")

    heights = []
    for t in range(args.ticks):
        eng.clear_external()
        eng.set_input(np.zeros(brain.encoder.n_in, np.float32))
        eng.set_external(cmd, np.ones(len(cmd), np.float32))   # 点亮"站起来"
        eng.step()
        forces = brain.forces()
        body.step(forces)
        s = body.state()
        heights.append(s["height"])
        if (t + 1) % max(1, args.ticks // 6) == 0:
            print(f"  t={t + 1:3d}  高度 {s['height']:.3f}  倾角 "
                  f"{s['roll']:+.3f}/{s['pitch']:+.3f}  力度均值 {forces.mean():.2f}")
    print(f"\n高度：起 {heights[0]:.3f} → 末 {heights[-1]:.3f}；"
          f"末段均值 {np.mean(heights[-20:]):.3f}；摔倒 {body.fallen()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
