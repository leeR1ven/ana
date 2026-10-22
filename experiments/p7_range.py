# -*- coding: utf-8 -*-
"""P7：用"思想范围"来测，而不是数命中几个单元。

做法：
  1. 训练 A→B→C 的顺序；
  2. 只点 A，看点亮的那一片**更像 A、B 还是 C**（重合度）；
  3. 用"集中度 = 最大重合 / 平均重合"看它是不是聚焦在一个范围，而不是散掉。

对照组：同样点 A，但不学习。

    python experiments/p7_range.py --ticks 800 --seeds 3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import Cortex, PlasticConfig, PlasticRule, build, build_spec   # noqa: E402
from experiments.p1_recall import run_obs, scene_obs                    # noqa: E402


def trial(ticks: int, net_seed: int, learn: bool) -> dict:
    sp = build_spec("xl", pfc_topk=0.15, pfc_adapt=0.0, seed=7 + net_seed)
    brain = build(sp, verbose=False)
    eng = brain.engine
    rule = PlasticRule(PlasticConfig(
        enabled=learn, eta=0.08, eta_sim=0.0, topk_out=32,
        full_cross=2_000_000, max_pairs=200_000,
        settle_overlap=0.9, settle_ticks=2, novelty_window=2,
        flush_every=50, flush_edges=200_000, decay_every=1000, decay=0.995,
        sharpen=1.0, consolidate=True, reuse_first=True), seed=3 + net_seed)
    ctx = Cortex(brain, plastic=rule)
    cyc = [1, 2, 3, 2]
    n = ticks if learn else min(ticks, 40)
    for t in range(n):
        ctx.tick(scene_obs(sp, cyc[(t // 20) % 4]), record=False)
    rule.flush(eng)

    run_obs(brain, scene_obs(sp, 1)); A = set(brain.pfc_code().tolist())
    run_obs(brain, scene_obs(sp, 2)); B = set(brain.pfc_code().tolist())
    run_obs(brain, scene_obs(sp, 3)); C = set(brain.pfc_code().tolist())

    def thought(ticks_on: int = 14):
        eng.act[:] = 0.0
        eng.reset_adaptation()
        for _ in range(ticks_on):
            eng.clear_external()
            eng.set_input(brain.encoder.encode(scene_obs(sp, 2)))   # 一直看着 B
            eng.set_external(list(A), np.ones(len(A), np.float32))  # 同时点着 A
            eng.step()
        return set(brain.pfc_code().tolist())

    got = thought()
    j = {k: len(got & v) / max(1, len(got | v)) for k, v in (("A", A), ("B", B), ("C", C))}
    conc = max(j.values()) / max(1e-9, np.mean(list(j.values())))
    return dict(A=len(A), B=len(B), C=len(C), got=len(got), j=j, conc=float(conc),
                where=max(j, key=j.get))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=800)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args(argv)
    for learn in (True, False):
        rows = [trial(args.ticks, s, learn) for s in range(args.seeds)]
        print(f"\n{'学习' if learn else '不学习（对照）'}")
        for s, r in enumerate(rows):
            print(f"  seed {s}: 点亮 {r['got']:4d}  重合 A {r['j']['A']:.2f} "
                  f"B {r['j']['B']:.2f} C {r['j']['C']:.2f}  "
                  f"更像 {r['where']}  集中度 {r['conc']:.2f}")
        conc = np.mean([r["conc"] for r in rows])
        where = [r["where"] for r in rows]
        print(f"  平均集中度 {conc:.2f}；更像 B 的种子 {where.count('B')}/{len(where)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
