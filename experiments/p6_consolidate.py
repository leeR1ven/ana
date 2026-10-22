# -*- coding: utf-8 -*-
"""P6：固化验证——用得多的边改得慢，但不是不能改。

两段训练同一个网络：
  第一段记下每条可塑边的权重和"用过几次"；
  第二段再练，比较"老边"和"刚连上的边"各自变了多少。

    python experiments/p6_consolidate.py --ticks 300
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import Cortex, PlasticConfig, PlasticRule, build, build_spec   # noqa: E402
from experiments.p1_recall import scene_obs                             # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=300)
    p.add_argument("--preset", default="xl")
    args = p.parse_args(argv)

    sp = build_spec(args.preset, pfc_topk=0.15, pfc_adapt=0.0)
    brain = build(sp, verbose=False)
    eng = brain.engine
    rule = PlasticRule(PlasticConfig(
        enabled=True, eta=0.08, eta_sim=0.0, topk_out=32,
        full_cross=2_000_000, max_pairs=200_000,
        settle_overlap=0.9, settle_ticks=2, novelty_window=2,
        flush_every=50, flush_edges=200_000, decay_every=1000, decay=0.995,
        sharpen=1.0, in_floor=64.0, pmi_floor=0.6, pmi_steep=0.5,
        consolidate=True, use_ref=3.0, long_term_at=8.0), seed=3)
    ctx = Cortex(brain, plastic=rule)
    cyc = [1, 2, 3, 2]

    def train(n, offset=0):
        for t in range(n):
            ctx.tick(scene_obs(sp, cyc[((t + offset) // 20) % 4]), record=False)
        rule.flush(eng)

    train(args.ticks)
    m = eng.group == 40
    w1 = eng.w[m].copy()
    key1 = eng.src[m].astype(np.int64) * eng.n + eng.dst[m]
    use1 = rule._use[:len(w1)].copy()
    n_lt = int((use1 >= rule.cfg.long_term_at).sum())
    print(f"第一段后：边 {len(w1)}，长期记忆 {n_lt}，平均用过 {use1.mean():.1f} 次")

    train(args.ticks, args.ticks)
    m2 = eng.group == 40
    w2 = eng.w[m2]
    key2 = eng.src[m2].astype(np.int64) * eng.n + eng.dst[m2]
    order = np.argsort(key2)
    pos = np.minimum(np.searchsorted(key2[order], key1), len(key2) - 1)
    same = key2[order][pos] == key1
    w2a = np.full(len(key1), np.nan, np.float32)
    w2a[same] = w2[order][pos[same]]
    delta = np.abs(w2a - w1)
    lt = (use1 >= rule.cfg.long_term_at) & same & ~np.isnan(w2a)
    fresh = (use1 <= 1) & same & ~np.isnan(w2a)
    d_lt = float(np.nanmean(delta[lt])) if lt.any() else 0.0
    d_fr = float(np.nanmean(delta[fresh])) if fresh.any() else 0.0
    print(f"第二段后：边 {int(m2.sum())}，长期记忆 {int(rule._frozen.sum())}")
    print(f"长期记忆边 {int(lt.sum())} 条：平均变化 {d_lt:.5f}")
    print(f"刚连上的边 {int(fresh.sum())} 条：平均变化 {d_fr:.5f}")
    print(f"结论：{'长期记忆明显改得慢' if d_lt < d_fr else '长期记忆没有变慢'}"
          f"（{d_lt:.5f} vs {d_fr:.5f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
