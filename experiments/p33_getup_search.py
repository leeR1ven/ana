# -*- coding: utf-8 -*-
"""起身专项搜索：随机模型 + 并行评估 getup2，保存最高的那个。

用法（系统 Python，需 mujoco）：
    python experiments/p33_getup_search.py --n 200 --jobs 14
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
        r = run_skill(kw, "getup2", ticks)
    except Exception:
        return kw, 0.0, 0.0
    return kw, float(r.get("height", 0.0)), float(r.get("dx", 0.0))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--jobs", type=int, default=0)
    p.add_argument("--ticks", type=int, default=140)
    a = p.parse_args(argv)
    jobs = a.jobs or max(1, (os.cpu_count() or 4) - 1)

    path = os.path.join(ROOT, "getup_best.json")
    best_kw, best_h = None, -1.0
    if os.path.exists(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
            best_kw, best_h = d["kw"], float(d["height"])
            print("已有最好高度 %.3f，只在更好时覆盖" % best_h, flush=True)
        except Exception:
            pass

    rng = np.random.default_rng(3)
    kws = [(dict(seed=int(rng.integers(1, 10 ** 6)),
                 k_near=int(rng.integers(4, 25)),
                 region_links=int(rng.integers(0, 129)),
                 w_lo=float(rng.uniform(0.02, 0.5)),
                 w_hi=float(rng.uniform(0.3, 2.0)),
                 posture_offset=float(rng.uniform(0.3, 1.2))), a.ticks)
           for _ in range(a.n)]

    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for kw, h, _ in ex.map(_work, kws, chunksize=1):
            done += 1
            if h > best_h:
                best_kw, best_h = kw, h
                print("  #%d 新高高度 %.3f" % (done, h), flush=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"height": best_h, "kw": best_kw}, f,
                              ensure_ascii=False, indent=2)
            elif done % 50 == 0:
                print("  #%d 当前最好 %.3f" % (done, best_h), flush=True)

    print("样本 %d 最高高度 %.3f" % (a.n, best_h))
    print("最好参数:", best_kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
