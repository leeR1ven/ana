# -*- coding: utf-8 -*-
"""P6：固化验证——老边"每次被用"改得少，不是"总变化少"。

两段训练同一个网络，比较每条边**每次被用到时**平均改了多少：
  老边（用过很多次）应该明显小于新边。

    python experiments/p6_consolidate.py --ticks 1200
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
    p.add_argument("--ticks", type=int, default=600)
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
        consolidate=True, reuse_first=True, min_out_new=4,
        use_ref=3.0, long_term_at=8.0), seed=3)
    ctx = Cortex(brain, plastic=rule)
    cyc = [1, 2, 3, 2]

    def train(n, offset=0):
        for t in range(n):
            ctx.tick(scene_obs(sp, cyc[((t + offset) // 20) % 4]), record=False)
        rule.flush(eng)

    def snap():
        m = eng.group == 40
        return (eng.w[m].copy(), eng.src[m].astype(np.int64) * eng.n + eng.dst[m],
                rule._use[:int(m.sum())].copy())

    train(args.ticks)
    w1, key1, use1 = snap()
    lt_at = use1 >= rule.cfg.long_term_at
    print(f"第一段后：边 {len(w1)}，长期记忆 {int(lt_at.sum())}，平均用过 {use1.mean():.1f} 次")

    train(args.ticks, args.ticks)
    w2, key2, use2 = snap()
    order = np.argsort(key2)
    pos = np.minimum(np.searchsorted(key2[order], key1), len(key2) - 1)
    same = key2[order][pos] == key1
    w2a = np.full(len(key1), np.nan, np.float32)
    u2a = np.zeros(len(key1), np.float32)
    w2a[same] = w2[order][pos[same]]
    u2a[same] = use2[order][pos[same]]
    delta = np.abs(w2a - w1)
    duse = u2a - use1
    rate = np.where(duse > 0, delta / np.maximum(duse, 1e-6), np.nan)

    lt = (use1 >= rule.cfg.long_term_at) & same & (duse > 0) & ~np.isnan(rate)
    fresh = (use1 <= 1) & same & (duse > 0) & ~np.isnan(rate)
    print(f"第二段后：边 {int((eng.group == 40).sum())}，长期记忆 {int(rule._frozen.sum())}")
    print(f"老边 {int(lt.sum())} 条：总变化 {np.nanmean(delta[lt]) if lt.any() else 0:.4f}，"
          f"每次被用改 {np.nanmean(rate[lt]) if lt.any() else 0:.6f}")
    print(f"新边 {int(fresh.sum())} 条：总变化 {np.nanmean(delta[fresh]) if fresh.any() else 0:.4f}，"
          f"每次被用改 {np.nanmean(rate[fresh]) if fresh.any() else 0:.6f}")
    if lt.any() and fresh.any():
        r_lt, r_fr = float(np.nanmean(rate[lt])), float(np.nanmean(rate[fresh]))
        print(f"结论：{'老边每次改得更少（难改成立）' if r_lt < r_fr else '老边没有更难改'} "
              f"（{r_lt:.6f} vs {r_fr:.6f}，比值 {r_lt / max(r_fr, 1e-9):.2f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
