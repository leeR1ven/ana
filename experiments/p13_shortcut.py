# -*- coding: utf-8 -*-
"""P13：区与区之间的"学出来的短路"。

规则（简单版）：两个相邻区的 L5 同时有单元在亮 → 从亮的视觉单元往亮的听觉单元
长一小批直连（权重小、可塑）。之后：
  · 只点视觉时，听觉应该比"没有短路"时更容易被带起来；
  · 这条通路不经过前额叶，所以是"短路"。

    python experiments/p13_shortcut.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import Cortex, PlasticConfig, PlasticRule, build, build_spec   # noqa: E402
from experiments.p1_recall import scene_obs                             # noqa: E402

GROUP = 58


def l5_active(eng, ids):
    a = eng.act[ids]
    return ids[a > 0.3]


def learn_shortcuts(eng, ids_a, ids_b, rng, rounds=60, k=2, w=0.05):
    n = 0
    for _ in range(rounds):
        a = l5_active(eng, ids_a)
        b = l5_active(eng, ids_b)
        if len(a) and len(b):
            for _k in range(k):
                sa = a[rng.integers(0, len(a), min(8, len(a)))]
                sb = b[rng.integers(0, len(b), min(8, len(b)))]
                eng.add_edges(sb.astype(np.int32), sa.astype(np.int32),
                              np.full(len(sa), w, np.float32), group=GROUP)
                n += len(sa)
    if n:
        eng.rebuild_backend()
    return n


def cue_and_measure(eng, cue_ids, target_ids, ticks=4):
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(np.zeros(eng.input_size, np.float32))
        eng.set_external(cue_ids, np.ones(len(cue_ids), np.float32))
        eng.step()
    return float(eng.act[target_ids].mean())


def main() -> int:
    sp = build_spec("full")
    brain = build(sp, verbose=False)
    eng = brain.engine
    va = brain.layers["vision.L5"].ids
    au = brain.layers["audio.L5"].ids

    def obs(seed):
        o = scene_obs(sp, seed)
        r = np.random.default_rng(seed + 90)
        o.audio = r.random((sp.audio_bands, sp.audio_frames)).astype(np.float32)
        return o

    ctx = Cortex(brain)
    for t in range(160):
        ctx.tick(obs([1, 2, 3, 2][(t // 20) % 4]), record=False)

    base = cue_and_measure(eng, l5_active(eng, va)[:40], au)
    rng = np.random.default_rng(3)
    n = learn_shortcuts(eng, va, au, rng)
    after = cue_and_measure(eng, l5_active(eng, va)[:40], au)
    print(f"短路边 {n} 条；点视觉时听觉 L5 平均活动：无短路 {base:.3f} → 有短路 {after:.3f}")
    print(f"结论：{'短路能传递（绕开前额叶）' if after > base + 0.02 else '没看出传递'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
