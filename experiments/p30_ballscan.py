# -*- coding: utf-8 -*-
"""搜索"看球走"：以 |dy_side|（左右球位造成的横向位移差）为目标。

用法（系统 Python，需 mujoco）：
    python experiments/p30_ballscan.py --n 200 --jobs 12
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

from agi.dog_tasks import run_skill   # noqa: E402


def _work(args):
    kw, ticks = args
    try:
        r = run_skill(kw, "walk_to_ball", ticks)
    except Exception:
        return kw, 0.0, 1.0
    return kw, abs(r.get("dy_side", 0.0)), float(abs(r.get("dx", 0.0)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--jobs", type=int, default=0)
    p.add_argument("--ticks", type=int, default=80)
    a = p.parse_args(argv)
    jobs = a.jobs or max(1, (os.cpu_count() or 4) - 1)

    path = os.path.join(ROOT, "ball_best.json")
    best_kw, best_dy = None, -1.0
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            best_kw, best_dy = d["kw"], float(d["dy_side"])
            print(f"已有最好 dy_side={best_dy:.4f}，只在更好时覆盖", flush=True)
        except Exception:
            pass

    rng = np.random.default_rng(11)
    kws = [(dict(seed=int(rng.integers(1, 10 ** 6)),
                 k_near=int(rng.integers(4, 25)),
                 region_links=int(rng.integers(0, 129)),
                 w_lo=float(rng.uniform(0.02, 0.5)),
                 w_hi=float(rng.uniform(0.3, 2.0))), a.ticks) for _ in range(a.n)]
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for kw, dy, dx in ex.map(_work, kws, chunksize=1):
            done += 1
            if dy > best_dy:
                best_kw, best_dy = kw, dy
                print(f"  #{done} 新高 |dy_side| {dy:.4f} dx {dx:.3f}", flush=True)
            elif done % 50 == 0:
                print(f"  #{done} 当前最好 {best_dy:.4f}", flush=True)
    print(f"样本 {a.n}，最大 |dy_side| {best_dy:.4f}")
    print("最好模型:", best_kw)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dy_side": best_dy, "kw": best_kw}, f, ensure_ascii=False, indent=2)
    print("已写", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
