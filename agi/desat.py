# -*- coding: utf-8 -*-
"""去饱和：把"整层几乎全亮"的层，门槛往上调到合理稀疏度。

为什么需要：小区的驱动分布退化（很多单元驱动相近），按分位定门槛会落在极低处，
整层几乎全过线（实测 balance.L4 40/40 全亮、audio.L5 639/640）。
整层饱和之后，输入再怎么变也传不出差异——小测里表现为"变化传不到这一层"。

做法：只对"活跃比例 > high"的层做二分搜索，把比例压到 target 附近；
其余层一律不动（避免影响已经正常的部分）。
"""
from __future__ import annotations

import numpy as np

from .encoding import Observation


def _probe(brain, spec, ticks: int = 10):
    """跑一张标准图，返回各层活跃比例。"""
    eng = brain.engine
    px = spec.px
    img = np.full((px, px, 3), 0.34, np.float32)
    img[px // 3: 2 * px // 3, px // 3: 2 * px // 3] = 0.67
    obs = Observation(
        vision=img,
        audio=np.full(spec.audio_bands * spec.audio_frames, 0.34, np.float32),
        touch=np.full(spec.touch_ch, 0.34, np.float32),
        proprio=np.full(spec.proprio_ch, 0.34, np.float32),
        balance=np.full(spec.balance_ch, 0.34, np.float32))
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(obs))
        eng.step()
    a = np.asarray(eng.act).copy()
    return {name: float((a[L.ids] > 0.5).mean()) for name, L in brain.layers.items()}


def desaturate(brain, spec, high: float = 0.70, target: float = 0.35,
               steps: int = 8) -> list[tuple[str, float, float]]:
    """把过亮的层调稀疏；返回 [(层名, 旧比例, 新比例)]。"""
    eng = brain.engine
    changed = []
    for name, L in brain.layers.items():
        frac = _probe(brain, spec)[name]
        if frac <= high:
            continue
        ids = L.ids
        base = eng.thr[ids].copy()
        lo, hi = 1.0, 8.0           # 门槛倍数
        best = base.copy()
        best_frac = frac
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            eng.set_thr(ids, thr=(base * mid).astype(np.float32),
                        soft=(np.maximum(base * mid, 1e-4) * 0.3).astype(np.float32))
            f = _probe(brain, spec)[name]
            if f > target:
                lo = mid
            else:
                hi = mid
            if abs(f - target) < abs(best_frac - target):
                best = (base * mid).astype(np.float32)
                best_frac = f
        eng.set_thr(ids, thr=best,
                    soft=(np.maximum(best, 1e-4) * 0.3).astype(np.float32))
        changed.append((name, frac, best_frac))
    return changed
