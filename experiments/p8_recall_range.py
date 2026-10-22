# -*- coding: utf-8 -*-
"""P8：只给提示、不给画面——"由 A 想到 B 的范围"。

协议：
  · 训练 A→B→C 的顺序（A、B、C 是三个结构上很不同的画面）；
  · 测试时**画面全黑**，只点 A；
  · 看想到的那一片里有多少落在 B 的范围内；再用"点同样多的随机单元"做对照；
  · 同时跑"不学习"的对照，扣除 A、B 码本身的共享。

判据（用户原话）：看激活的大多数字有没有形成大概的思想范围，而不是数单个单元。

    python experiments/p8_recall_range.py --seeds 5
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


def trial(seed: int, ticks: int, learn: bool) -> tuple[int, float, float]:
    sp = build_spec("xl", pfc_topk=0.15, pfc_adapt=0.0, seed=7 + seed)
    brain = build(sp, verbose=False)
    eng = brain.engine
    rule = PlasticRule(PlasticConfig(
        enabled=learn, eta=0.08, eta_sim=0.0, topk_out=32,
        full_cross=2_000_000, max_pairs=200_000,
        settle_overlap=0.9, settle_ticks=2, novelty_window=2,
        flush_every=50, flush_edges=200_000, decay_every=1000, decay=0.995,
        sharpen=1.0, consolidate=True, reuse_first=True), seed=3 + seed)
    ctx = Cortex(brain, plastic=rule)
    cyc = [1, 2, 3, 2]
    for t in range(ticks if learn else 40):
        ctx.tick(scene_obs(sp, cyc[(t // 20) % 4]), record=False)
    rule.flush(eng)

    run_obs(brain, scene_obs(sp, 1)); A = set(brain.pfc_code().tolist())
    run_obs(brain, scene_obs(sp, 2)); B = set(brain.pfc_code().tolist())

    def think(cue_ids, n: int = 12):
        eng.act[:] = 0.0
        eng.reset_adaptation()
        for _ in range(n):
            eng.clear_external()
            eng.set_input(np.zeros(brain.encoder.n_in, np.float32))   # 不给画面
            eng.set_external(cue_ids, np.ones(len(cue_ids), np.float32))
            eng.step()
        return set(brain.pfc_code().tolist())

    rng = np.random.default_rng(seed + 11)
    gotA = think(list(A))
    gotR = think(list(rng.choice(brain.pfc_ids, len(A), replace=False)))
    return (len(gotA),
            len(gotA & B) / max(1, len(B)),
            len(gotR & B) / max(1, len(B)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--ticks", type=int, default=800)
    args = p.parse_args(argv)
    for learn in (True, False):
        rows = [trial(s, args.ticks, learn) for s in range(args.seeds)]
        hit = np.array([x[1] for x in rows])
        rnd = np.array([x[2] for x in rows])
        print(f"{'学习' if learn else '对照'}: "
              + "  ".join(f"{h:.2f}/{r:.2f}" for h, r in zip(hit, rnd)))
        print(f"   点A含B 均值 {hit.mean():.3f}；随机含B {rnd.mean():.3f}；"
              f"差 {(hit - rnd).mean():+.3f}；每个种子的差 "
              + " ".join(f"{(h - r):+.2f}" for h, r in zip(hit, rnd)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
