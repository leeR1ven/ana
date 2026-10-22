# -*- coding: utf-8 -*-
"""并行走路筛选：把机器核数用满（多进程），比串行快数倍。

用法（系统 Python，需 mujoco）：
    python experiments/p29_walk_par.py --n 400 --jobs 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.p27_walk_sweep import one   # noqa: E402


def _work(args):
    kw, ticks = args
    r = one(kw, ticks)
    return (kw, None if r is None else abs(r["dx"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--jobs", type=int, default=0)
    p.add_argument("--ticks", type=int, default=100)
    p.add_argument("--seed-kw", default="", help="从 walk_best.json 起，做局部搜索")
    a = p.parse_args(argv)
    jobs = a.jobs or max(1, (os.cpu_count() or 4) - 1)

    rng = np.random.default_rng(7)
    base = None
    if a.seed_kw:
        with open(a.seed_kw, encoding="utf-8") as f:
            base = json.load(f)["kw"]

    def make_kw(i):
        if base is None:
            return dict(seed=int(rng.integers(1, 10 ** 6)),
                        k_near=int(rng.integers(4, 25)),
                        region_links=int(rng.integers(0, 129)),
                        w_lo=float(rng.uniform(0.02, 0.5)),
                        w_hi=float(rng.uniform(0.3, 2.0)))
        kw = dict(base)
        kw["seed"] = int(rng.integers(1, 10 ** 6))
        kw["k_near"] = int(np.clip(kw["k_near"] + rng.integers(-2, 3), 4, 24))
        kw["region_links"] = int(np.clip(kw["region_links"] + rng.integers(-16, 17), 0, 128))
        for k, (lo, hi) in (("w_lo", (0.02, 0.5)), ("w_hi", (0.3, 2.0))):
            kw[k] = float(np.clip(kw[k] * rng.uniform(0.8, 1.25), lo, hi))
        return kw

    kws = [(make_kw(i), a.ticks) for i in range(a.n)]
    best_kw, best_dx = None, -1.0
    prev_path = os.path.join(ROOT, "walk_best.json")
    if os.path.exists(prev_path):          # 只有更好才覆盖，别把好模型冲掉
        try:
            with open(prev_path, encoding="utf-8") as f:
                _p = json.load(f)
            best_dx, best_kw = float(_p["dx"]), _p["kw"]
            print(f"已有最好 {best_dx:.3f}，只在更好时覆盖", flush=True)
        except Exception:
            pass
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for kw, dx in ex.map(_work, kws, chunksize=1):
            done += 1
            if dx is not None and dx > best_dx:
                best_kw, best_dx = kw, dx
                print(f"  #{done} 新高 |位移| {dx:.3f}", flush=True)
            elif done % 50 == 0:
                print(f"  #{done} 当前最好 {best_dx:.3f}", flush=True)
    print(f"并行 {jobs} 进程，样本 {a.n}，最大 |位移| {best_dx:.3f}")
    print("最好模型:", best_kw)
    out = {"dx": best_dx, "fallen": False, "kw": best_kw}
    path = os.path.join(ROOT, "walk_best.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("已写", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
