# -*- coding: utf-8 -*-
"""只给 A、不给画面：点 A 想到的范围，是否比"随机点一片"更接近 B。"""
import sys
sys.path.insert(0, "ana")
import numpy as np
from agi import Cortex, PlasticConfig, PlasticRule, build, build_spec
from experiments.p1_recall import run_obs, scene_obs


def trial(seed, learn=True):
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
    for t in range(800 if learn else 40):
        ctx.tick(scene_obs(sp, cyc[(t // 20) % 4]), record=False)
    rule.flush(eng)

    run_obs(brain, scene_obs(sp, 1)); A = set(brain.pfc_code().tolist())
    run_obs(brain, scene_obs(sp, 2)); B = set(brain.pfc_code().tolist())

    def think(cue_ids, n=12):
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
    fA = len(gotA & B) / max(1, len(B))
    fR = len(gotR & B) / max(1, len(B))
    return len(gotA), fA, fR


for learn in (True, False):
    rows = [trial(s, learn) for s in range(3)]
    print(f"{'学习' if learn else '对照'}: " + "  ".join(
        f"seed{i} 想到{got}个 含B {a:.2f} 随机含B {r:.2f} 差 {a - r:+.2f}"
        for i, (got, a, r) in enumerate(rows)))
    print(f"   平均：点A含B {np.mean([x[1] for x in rows]):.3f}  "
          f"随机含B {np.mean([x[2] for x in rows]):.3f}  "
          f"差 {np.mean([x[1] - x[2] for x in rows]):+.3f}")
