# -*- coding: utf-8 -*-
"""P1 联想（稳定码学习版）。

协议（按第七组实验定下来的）：
  · 每个场景保持 20 拍，让码稳定；
  · 只在"稳定码之间"学，并且只把"最近两段里没出现过"的单元当目标；
  · 测法：一直看 B，同时一直点着 A（或 C 做对照），比较 B 的专属单元活动。

判据：点 A 时 B 专属单元的活动 > 点 C 时（A 是跟着 B 出现的，C 不是）。

    python experiments/p1_assoc.py --ticks 800 --gain 4 --seeds 3
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


def trial(ticks: int, gain: float, net_seed: int, hold: int = 20, learn: bool = True,
          pmi: bool = True, in_floor: float = 64.0, eta: float = 0.08) -> dict:
    sp = build_spec("xl", pfc_topk=0.15, pfc_adapt=0.0, seed=7 + net_seed)
    brain = build(sp, verbose=False)
    eng = brain.engine
    rule = PlasticRule(PlasticConfig(
        enabled=learn, eta=0.08, eta_sim=0.0, topk_out=32,
        full_cross=2_000_000, max_pairs=200_000,
        settle_overlap=0.9, settle_ticks=2, novelty_window=2,
        flush_every=50, flush_edges=200_000, decay_every=1000, decay=0.995,
        sharpen=1.0, in_floor=in_floor, pmi_gate=pmi,
        pmi_floor=0.6, pmi_steep=0.5,
        prop_gain=gain), seed=3 + net_seed)
    ctx = Cortex(brain, plastic=rule)
    cyc = [1, 2, 3, 2]
    for t in range(ticks if learn else min(ticks, 40)):
        ctx.tick(scene_obs(sp, cyc[(t // hold) % 4]), record=False)
    rule.flush(eng)

    run_obs(brain, scene_obs(sp, 1)); A = set(brain.pfc_code().tolist())
    run_obs(brain, scene_obs(sp, 2)); B = set(brain.pfc_code().tolist())
    run_obs(brain, scene_obs(sp, 3)); C = set(brain.pfc_code().tolist())
    Bonly = np.array(sorted((B - C) - A), int)

    def see(cue_ids=None, n: int = 14):
        eng.act[:] = 0.0
        eng.reset_adaptation()
        for _ in range(n):
            eng.clear_external()
            eng.set_input(brain.encoder.encode(scene_obs(sp, 2)))
            if cue_ids:
                eng.set_external(cue_ids, np.ones(len(cue_ids), np.float32))
            eng.step()
        return eng.act_np.copy()

    a0, aA, aC = see(), see(list(A)), see(list(C))
    return dict(bonly=len(Bonly),
                base=float(a0[Bonly].mean()),
                cueA=float(aA[Bonly].mean()),
                cueC=float(aC[Bonly].mean()),
                spec=float(aA[Bonly].mean() - aC[Bonly].mean()),
                edges=int((eng.group == 40).sum()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=800)
    p.add_argument("--gain", type=float, default=4.0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--control", action="store_true", help="不学习（对照）")
    p.add_argument("--no-pmi", action="store_true")
    p.add_argument("--no-hub", action="store_true")
    p.add_argument("--eta", type=float, default=0.08)
    args = p.parse_args(argv)
    res = []
    for s in range(args.seeds):
        r = trial(args.ticks, args.gain, s, learn=not args.control,
                  pmi=not args.no_pmi,
                  in_floor=1e9 if args.no_hub else 64.0, eta=args.eta)
        res.append(r)
        print(f"seed {s}: B独有 {r['bonly']:4d}  只看B {r['base']:.3f}  "
              f"带A {r['cueA']:.3f}  带C {r['cueC']:.3f}  "
              f"特异性 {r['spec']:+.3f}  边 {r['edges']}")
    spec = np.array([r["spec"] for r in res])
    print(f"\n特异性 均值 {spec.mean():+.3f}  中位 {np.median(spec):+.3f}  "
          f">0 的种子 {int((spec > 0).sum())}/{len(spec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
