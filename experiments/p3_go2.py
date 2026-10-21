# -*- coding: utf-8 -*-
"""P3：把大脑接到宇树 Go2（MuJoCo）上跑闭环。

这一版只验证三件事：
  1. 50 Hz 的闭环跑得动（大脑 + 物理，每拍 < 20 ms）；
  2. 狗在"大脑也在跑"的情况下仍然站得住（PD 撑底，力度当小幅偏置）；
  3. 感觉确实进了大脑（本体/平衡/触地的活动能看到）。

    python experiments/p3_go2.py --ticks 300 --preset full
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec                                        # noqa: E402
from agi.body_go2 import Go2Body                                        # noqa: E402
from agi.loop import Cortex                                             # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=300)
    p.add_argument("--preset", default="full")
    p.add_argument("--posture-offset", type=float, default=0.02,
                   help="力度档 → 目标关节角的最大偏置（弧度）")
    args = p.parse_args(argv)

    sp = build_spec(args.preset, pfc_topk=0.005, pfc_adapt=0.01)
    t0 = time.perf_counter()
    brain = build(sp, verbose=False)
    print(f"建网 {time.perf_counter() - t0:.1f}s：{brain.engine.n:,} 神经元 / "
          f"{brain.engine.n_edges:,} 边")
    body = Go2Body(posture_offset=args.posture_offset)
    body.reset()
    ctx = Cortex(brain, trace=False)

    ts, heights, rolls, forces = [], [], [], []
    for i in range(args.ticks):
        obs = body.observe(sp)
        t1 = time.perf_counter()
        tk = ctx.tick(obs, record=False)
        body.step(tk.forces)
        ts.append((time.perf_counter() - t1) * 1e3)
        s = body.state()
        heights.append(s["height"])
        rolls.append(s["roll"])
        forces.append(tk.forces.copy())
        if (i + 1) % max(1, args.ticks // 6) == 0:
            print(f"  t={i + 1:4d}  高度 {s['height']:.3f}  roll {s['roll']:+.3f}  "
                  f"pitch {s['pitch']:+.3f}  前额叶 {tk.pfc_active:4d}  "
                  f"{ts[-1]:.1f} ms")
    ts = np.asarray(ts)
    F = np.stack(forces)
    print(f"\n闭环 {args.ticks} 拍：每拍 均值 {ts.mean():.1f} ms / p95 "
          f"{np.percentile(ts, 95):.1f} ms（预算 20）")
    print(f"高度 最小 {np.min(heights):.3f} / 末值 {heights[-1]:.3f}；"
          f"最大倾角 {np.max(np.abs(rolls)):.3f} rad；摔倒 {body.fallen()}")
    print(f"出力 均值 {F.mean():.2f}，非零比例 {(F > 0.05).mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
