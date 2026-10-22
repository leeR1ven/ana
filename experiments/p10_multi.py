# -*- coding: utf-8 -*-
"""P10：教第二个动作，第一个不被冲掉；已经会的就不再重复写边（门控生长）。

  1. 看 A → 教"站"（每块肌肉 5 档 → 力度 0.5）；
  2. 看 B → 教"用力撑"（7 档 → 力度 0.7）；
  3. 回头点 A：还应是 0.5；点 B：0.7；
  4. 再教一次 A：应该"回忆已经对了"，0 条新边。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec                                        # noqa: E402
from experiments.p1_recall import run_obs, scene_obs                    # noqa: E402

G_ONESHOT = 60


def teach(eng, src_ids, force_cells, levels: int, w: float = 6.0) -> int:
    src, dst = [], []
    for m in range(len(force_cells)):
        src.append(np.repeat(src_ids, levels))
        dst.append(np.tile(force_cells[m][:levels], len(src_ids)))
    eng.add_edges(np.concatenate(src), np.concatenate(dst),
                  np.full(sum(len(s) for s in src), w, np.float32), group=G_ONESHOT)
    eng.rebuild_backend()
    return int(sum(len(s) for s in src))


def think(eng, brain, cue, ticks: int = 12):
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(np.zeros(brain.encoder.n_in, np.float32))
        eng.set_external(cue, np.ones(len(cue), np.float32))
        eng.step()
    return brain.forces().copy()


def main() -> int:
    sp = build_spec("full")
    brain = build(sp, verbose=False)
    eng = brain.engine
    run_obs(brain, scene_obs(sp, 1)); Aall = set(brain.pfc_code().tolist())
    run_obs(brain, scene_obs(sp, 2)); Ball = set(brain.pfc_code().tolist())
    # 只教"这个场面独有"的单元，否则两个动作会互相串（共享单元会同时驱动两套边）
    A = np.array(sorted(Aall - Ball), int)
    B = np.array(sorted(Ball - Aall), int)

    print(f"教 A（站，5 档）：写入 {teach(eng, A, brain.motor['force'], 5)} 条边")
    fA = think(eng, brain, A)
    print(f"  点 A → 力度 {fA.mean():.2f}")

    print(f"教 B（撑，7 档）：写入 {teach(eng, B, brain.motor['force'], 7)} 条边")
    fB = think(eng, brain, B)
    print(f"  点 B → 力度 {fB.mean():.2f}")

    fA2 = think(eng, brain, A)
    print(f"  再点 A → 力度 {fA2.mean():.2f}（教 B 之前是 {fA.mean():.2f}）")

    # 门控生长：先试着回忆，已经对了就不再写边
    pred = think(eng, brain, A).mean()
    if abs(pred - 0.5) < 0.12:
        added = 0
    else:
        added = teach(eng, A, brain.motor["force"], 5)
    print(f"门控：点 A 已能给出 {pred:.2f}，新增边 {added} 条")

    ok1 = abs(fA2.mean() - fA.mean()) < 0.02
    ok2 = abs(fB.mean() - 0.7) < 0.05
    print(f"结论：{'两个动作并存、互不冲掉' if (ok1 and ok2) else '有干扰'}；"
          f"门控 {'生效' if added == 0 else '未生效'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
