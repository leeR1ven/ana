# -*- coding: utf-8 -*-
"""走路筛选：随机生成大量模型，看谁的位移最大（在进程内跑，比子进程快）。

用法（系统 Python，需 mujoco）：
    python experiments/p27_walk_sweep.py --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec                      # noqa: E402
from agi.body_go2 import Go2Body                       # noqa: E402
from agi.encoding import Observation                   # noqa: E402


def one(kw, ticks=100, preset="small"):
    sp = build_spec(preset, seed=7 + kw["seed"])
    sp.k_near = kw["k_near"]
    sp.region_links = kw["region_links"]
    sp.w_lo = kw["w_lo"]
    sp.w_hi = kw["w_hi"]
    try:
        brain = build(sp, verbose=False)
    except Exception:
        return None
    eng = brain.engine
    body = Go2Body(posture_offset=0.12)
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    px = sp.px
    obs = Observation(vision=np.full((px, px, 3), 0.5, np.float32),
                      audio=np.full(sp.audio_bands * sp.audio_frames, 0.34, np.float32),
                      touch=np.full(sp.touch_ch, 0.34, np.float32),
                      proprio=np.full(sp.proprio_ch, 0.34, np.float32),
                      balance=np.full(sp.balance_ch, 0.34, np.float32))
    xs = []
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(obs))
        eng.step()
        body.step(brain.forces())
        xs.append(float(body.data.qpos[0]))
    return dict(dx=xs[-1] - x0, fallen=bool(body.fallen()), kw=kw)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--ticks", type=int, default=100)
    ap = p.parse_args(argv)
    rng = np.random.default_rng(0)
    best = None
    rows = []
    for i in range(ap.n):
        kw = dict(seed=int(rng.integers(1, 10 ** 6)),
                  k_near=int(rng.integers(4, 25)),
                  region_links=int(rng.integers(0, 129)),
                  w_lo=float(rng.uniform(0.02, 0.5)),
                  w_hi=float(rng.uniform(0.3, 2.0)))
        r = one(kw, ap.ticks)
        if r is None:
            continue
        rows.append(r)
        if best is None or abs(r["dx"]) > abs(best["dx"]):
            best = r
            print(f"  #{i+1} 新高 |位移| {abs(r['dx']):.3f} {kw}", flush=True)
        elif (i + 1) % 20 == 0:
            print(f"  #{i+1} 当前最好 |位移| {abs(best['dx']):.3f}", flush=True)
    if rows:
        ds = [abs(r["dx"]) for r in rows]
        print(f"样本 {len(rows)}  最大 |位移| {max(ds):.3f}  中位 {np.median(ds):.3f}  均值 {np.mean(ds):.3f}")
        with open(os.path.join(ROOT, "walk_best.json"), "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
        print("最好模型:", best["kw"], "位移 %.3f" % best["dx"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
