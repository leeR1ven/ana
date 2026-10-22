import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agi import build_spec, build, Cortex, PlasticConfig, PlasticRule
from experiments.p1_recall import scene_obs, run_obs


def trial(learn):
    sp = build_spec("full")
    b = build(sp, verbose=False)
    eng = b.engine
    r = PlasticRule(PlasticConfig(
        enabled=learn, eta=0.08, eta_sim=0.0, topk_out=32,
        full_cross=2_000_000, max_pairs=200_000, min_change=0.3,
        flush_every=50, flush_edges=200_000, decay_every=1000, decay=0.995,
        sharpen=1.0, consolidate=True, reuse_first=True), seed=3)
    ctx = Cortex(b, plastic=r)
    cyc = [1, 2, 3, 2]
    for t in range(800):
        ctx.tick(scene_obs(sp, cyc[(t // 20) % 4]), record=False)
    r.flush(eng)
    run_obs(b, scene_obs(sp, 1)); A = set(b.pfc_code().tolist())
    run_obs(b, scene_obs(sp, 2)); B = set(b.pfc_code().tolist())
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(12):
        eng.clear_external()
        eng.set_input(np.zeros(b.encoder.n_in, np.float32))
        eng.set_external(list(A), np.ones(len(A), np.float32))
        eng.step()
    got = set(b.pfc_code().tolist())
    return len(got & B) / max(1, len(B)), int((eng.group == 40).sum())


for learn in (False, True):
    h, e = trial(learn)
    tag = "学习" if learn else "对照"
    print(f"{tag}: 含B {h:.2f}  联想边 {e}")
