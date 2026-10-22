# -*- coding: utf-8 -*-
"""走路迭代：以 walk_best.json 为种子，反复小步变异，保留位移更大的。

用法（系统 Python，需 mujoco）：
    python experiments/p28_walk_refine.py --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.p27_walk_sweep import one   # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--ticks", type=int, default=100)
    ap = p.parse_args(argv)
    path = os.path.join(ROOT, "walk_best.json")
    with open(path, encoding="utf-8") as f:
        cur = json.load(f)
    best_kw, best_dx = dict(cur["kw"]), abs(cur["dx"])
    print(f"起点 |位移| {best_dx:.3f} {best_kw}", flush=True)
    rng = np.random.default_rng(1)
    for i in range(ap.n):
        kw = dict(best_kw)
        kw["seed"] = int(rng.integers(1, 10 ** 6))
        kw["k_near"] = int(np.clip(kw["k_near"] + rng.integers(-2, 3), 4, 24))
        kw["region_links"] = int(np.clip(kw["region_links"] + rng.integers(-16, 17), 0, 128))
        kw["w_lo"] = float(np.clip(kw["w_lo"] * rng.uniform(0.8, 1.25), 0.01, 0.6))
        kw["w_hi"] = float(np.clip(kw["w_hi"] * rng.uniform(0.8, 1.25), 0.05, 2.0))
        r = one(kw, ap.ticks)
        if r is None:
            continue
        d = abs(r["dx"])
        if d > best_dx:
            best_dx, best_kw = d, kw
            print(f"  #{i+1} 新高 |位移| {d:.3f} {kw}", flush=True)
        elif (i + 1) % 25 == 0:
            print(f"  #{i+1} 当前最好 {best_dx:.3f}", flush=True)
    out = dict(dx=best_dx, fallen=False, kw=best_kw)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"最好 |位移| {best_dx:.3f} {best_kw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
