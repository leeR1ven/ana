# -*- coding: utf-8 -*-
"""P1 联想召回（可复现版）。

测法：训练后撤掉画面，**只点 A 的那些前额叶单元**，看 B 的"专属单元"
（B 有而 A 没有的）会不会亮；再点同样多的随机单元做对照。

    python experiments/p1_recall.py --ticks 800 --pmi on
    python experiments/p1_recall.py --ticks 800 --pmi off
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


def scene_obs(spec, seed: int) -> Observation:
    """三个结构上明显不同的画面（用随机纹理当 A/B/C 是错的：局部特征上它们统计一样）。"""
    px = spec.px
    if seed == 1:                       # 红球
        from agi.world import _blob
        img = np.zeros((px, px, 3), np.float32) + 0.05
        img = np.maximum(img, _blob(px, 0.3, 0.4, 0.2, (0.95, 0.2, 0.1)))
    elif seed == 2:                     # 随机噪声
        img = np.random.default_rng(5).random((px, px, 3)).astype(np.float32)
    else:                               # 棋盘
        g = np.indices((px, px)).sum(0) % 8
        img = np.repeat((g < 4)[..., None], 3, axis=2).astype(np.float32) * 0.9
    return Observation(vision=img.astype(np.float32),
                       audio=np.zeros((spec.audio_bands, spec.audio_frames), np.float32),
                       touch=np.full(spec.touch_ch, 0.3, np.float32),
                       proprio=np.full(spec.proprio_ch, 0.5, np.float32),
                       balance=np.full(spec.balance_ch, 0.5, np.float32))


def run_obs(brain, obs, ticks: int = 14) -> None:
    eng = brain.engine
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(obs))
        eng.step()


def cue(brain, ids, ticks: int = 10) -> set:
    eng = brain.engine
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(np.zeros(brain.encoder.n_in, np.float32))
        eng.set_external(ids, np.ones(len(ids), np.float32))
        eng.step()
    return set(brain.pfc_code().tolist())


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=800)
    p.add_argument("--preset", default="xl")
    p.add_argument("--pmi", default="on", choices=("on", "off"))
    p.add_argument("--net-seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=1)
    args = p.parse_args(argv)

    sp = build_spec(args.preset, pfc_topk=0.005, pfc_adapt=0.01,
                    seed=7 + args.net_seed)
    brain = build(sp, verbose=False)
    rule = PlasticRule(PlasticConfig(enabled=True, eta=0.10, eta_sim=0.02,
                                     topk_out=8, flush_every=50, flush_edges=20000,
                                     decay_every=1000, decay=0.995,
                                     pmi_gate=(args.pmi == "on")),
                       seed=3 + args.net_seed)
    ctx = Cortex(brain, plastic=rule)
    cyc = [1, 2, 3, 2]
    for t in range(args.ticks):
        ctx.tick(scene_obs(sp, cyc[(t // 8) % 4]), record=False)
    rule.flush(brain.engine)

    run_obs(brain, scene_obs(sp, 1))
    A = set(brain.pfc_code().tolist())
    run_obs(brain, scene_obs(sp, 2))
    B = set(brain.pfc_code().tolist())
    Buniq = B - A
    got = cue(brain, list(A))
    hits = len(Buniq & got)
    # 判别式指标：点 A 之后，落到 B 的多还是落到 C（没跟着 A 出现的）的多
    run_obs(brain, scene_obs(sp, 3))
    C = set(brain.pfc_code().tolist())
    hitB, hitC = len(B & got), len(C & got)
    rng = np.random.default_rng(args.eval_seed)
    rnd = [len(Buniq & cue(brain, list(rng.choice(brain.pfc_ids, len(A), replace=False))))
           for _ in range(6)]
    edges = int((brain.engine.group == 40).sum())
    print(f"pmi={args.pmi} 训练 {args.ticks} 拍: 边 {edges}, A {len(A)}, B {len(B)}, "
          f"B 专属 {len(Buniq)}, 点 A 命中 {hits}, 乱点 {np.mean(rnd):.1f} "
          f"(范围 {min(rnd)}-{max(rnd)}), 倍数 {hits / max(1e-9, np.mean(rnd)):.1f}")
    print(f"   判别式（只看专属单元）：点 A 后落到 B 专属 {len(Buniq & got)} / "
          f"落到 C 专属 {len((C - A) & got)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
