# -*- coding: utf-8 -*-
"""规模/耗时基准：决定"能做多大"的是边数，不是神经元数。"""
from __future__ import annotations

import time

import numpy as np


def bench_engine(brain, ticks: int = 20) -> dict:
    """只计 engine.step()（不含编码和读运动区）。"""
    eng = brain.engine
    from .world import sample_observation
    obs = sample_observation(brain.spec, np.random.default_rng(1), 0)
    eng.set_input(brain.encoder.encode(obs))
    for _ in range(3):
        eng.step()
    ts = []
    for _ in range(ticks):
        t0 = time.perf_counter()
        eng.step()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return dict(mean_ms=float(ts.mean()), p95_ms=float(np.percentile(ts, 95)),
                edges=eng.n_edges, neurons=eng.n)


def report(brain, ticks: int = 20) -> str:
    b = bench_engine(brain, ticks)
    sp = brain.spec
    budget = 1000.0 / sp.rate_hz
    per_step = budget / max(1, sp.substeps)
    lines = [
        f"神经元 {brain.engine.n:,}   边 {brain.engine.n_edges:,}   "
        f"后端 {brain.engine._backend}",
        f"一拍 {budget:.1f} ms（{sp.rate_hz:.0f} Hz，{sp.substeps} 小步/拍）",
        f"每小步 {b['mean_ms']:.2f} ms（p95 {b['p95_ms']:.2f}），"
        f"预算 {per_step:.2f} ms/小步 → {'够' if b['mean_ms'] * sp.substeps < budget else '超'}",
        f"等效 {brain.engine.n_edges / max(b['mean_ms'], 1e-9) / 1e3:.1f} M 边/秒",
    ]
    return "\n".join(lines)
