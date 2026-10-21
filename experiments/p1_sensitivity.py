# -*- coding: utf-8 -*-
"""P1 联想：参数敏感性 + 补全复测。

    python experiments/p1_sensitivity.py --seeds 3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import Cortex, PlasticConfig, PlasticRule, build, build_spec   # noqa: E402
from agi.encoding import Observation                                    # noqa: E402
from experiments.p1_recall import run_obs, scene_obs                    # noqa: E402


def faded(spec, k=0.3):
    o = scene_obs(spec, 2)
    return Observation(vision=(o.vision * k).astype(np.float32), audio=o.audio,
                       touch=o.touch, proprio=o.proprio, balance=o.balance)


def trial(gain, net_seed, ticks=800, hold=20):
    sp = build_spec("xl", pfc_topk=0.15, pfc_adapt=0.0, seed=7 + net_seed)
    brain = build(sp, verbose=False)
    eng = brain.engine
    rule = PlasticRule(PlasticConfig(
        enabled=True, eta=0.08, eta_sim=0.0, topk_out=32,
        full_cross=2_000_000, max_pairs=200_000,
        settle_overlap=0.9, settle_ticks=2, novelty_window=2,
        flush_every=50, flush_edges=200_000, decay_every=1000, decay=0.995,
        sharpen=1.0, in_floor=64.0, pmi_floor=0.6, pmi_steep=0.5,
        prop_gain=gain), seed=3 + net_seed)
    ctx = Cortex(brain, plastic=rule)
    cyc = [1, 2, 3, 2]
    for t in range(ticks):
        ctx.tick(scene_obs(sp, cyc[(t // hold) % 4]), record=False)
    rule.flush(eng)

    run_obs(brain, scene_obs(sp, 1)); A = set(brain.pfc_code().tolist())
    run_obs(brain, scene_obs(sp, 2)); B = set(brain.pfc_code().tolist())
    run_obs(brain, scene_obs(sp, 3)); C = set(brain.pfc_code().tolist())
    Bonly = np.array(sorted((B - C) - A), int)

    def see(cue_ids=None, obs=None, n=14):
        eng.act[:] = 0.0
        eng.reset_adaptation()
        o = obs if obs is not None else scene_obs(sp, 2)
        for _ in range(n):
            eng.clear_external()
            eng.set_input(brain.encoder.encode(o))
            if cue_ids:
                eng.set_external(cue_ids, np.ones(len(cue_ids), np.float32))
            eng.step()
        return eng.act_np.copy()

    aA = see(list(A))
    aC = see(list(C))
    spec = float(aA[Bonly].mean() - aC[Bonly].mean())
    # 补全：模糊 B（亮度 30%）先看 14 拍，再单独看 14 拍，比较 B 专属单元活动
    b0 = see(None, faded(sp))
    b1 = see(list(A), faded(sp))
    return spec, float(b1[Bonly].mean() - b0[Bonly].mean())


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args(argv)
    for gain in (2.0, 4.0, 8.0):
        specs, comps = [], []
        for s in range(args.seeds):
            sp_, cp_ = trial(gain, s)
            specs.append(sp_)
            comps.append(cp_)
        print(f"gain {gain}: 特异性 均值 {np.mean(specs):+.3f} "
              f"(各 {['%+.3f' % x for x in specs]})   "
              f"补全提升 均值 {np.mean(comps):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
