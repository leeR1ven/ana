# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "ana")
import numpy as np
from agi import Cortex, PlasticConfig, PlasticRule, build, build_spec
from experiments.p1_recall import run_obs, scene_obs


def trial(seed, inh):
    sp = build_spec("xl", pfc_topk=0.15, pfc_adapt=0.0, seed=7 + seed)
    b = build(sp, verbose=False)
    eng = b.engine
    r = PlasticRule(PlasticConfig(
        enabled=True, eta=0.08, eta_sim=0.0, topk_out=32, full_cross=2_000_000,
        max_pairs=200_000, settle_overlap=0.9, settle_ticks=2, novelty_window=2,
        flush_every=50, flush_edges=200_000, decay_every=1000, decay=0.995,
        sharpen=1.0, consolidate=True, reuse_first=True, inh_targets=inh), seed=3 + seed)
    ctx = Cortex(b, plastic=r)
    cyc = [1, 2, 3, 2]
    for t in range(800):
        ctx.tick(scene_obs(sp, cyc[(t // 20) % 4]), record=False)
    r.flush(eng)

    run_obs(b, scene_obs(sp, 1)); A = set(b.pfc_code().tolist())
    run_obs(b, scene_obs(sp, 2)); B = set(b.pfc_code().tolist())

    def think(cue, n=12):
        eng.act[:] = 0.0
        eng.reset_adaptation()
        for _ in range(n):
            eng.clear_external()
            eng.set_input(np.zeros(b.encoder.n_in, np.float32))
            eng.set_external(cue, np.ones(len(cue), np.float32))
            eng.step()
        return set(b.pfc_code().tolist())

    got = think(list(A))
    return len(got & B) / max(1, len(B)), int((eng.group == 41).sum())


for inh in (False, True):
    rows = [trial(s, inh) for s in range(2)]
    print(f"inh_targets={inh}（只罚'真没被驱动'）: "
          + "  ".join(f"含B {h:.2f}（抑制边 {n}）" for h, n in rows))
