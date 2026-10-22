# -*- coding: utf-8 -*-
"""演化搜索：照老库 chain-neuron-architecture/code/train_loop.py 的做法。

  · 随机生成一批"出生参数"（Spec 里的数值项）
  · 用固定题库打分（同一套题，不许换题）
  · 选优 → 精英保留 → 在最优者附近抖动 → 下一代

注意：这里**不用可塑性**。可塑性是"在一条命里学"，演化是"换一批命再选"，
老库靠的是后者。两者可以并存，但不要指望前者学会动作。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .spec import build_spec
from .build import build
from .loop import Cortex
from .encoding import Observation
from .world import _blob

HERE = os.path.dirname(os.path.abspath(__file__))

# 可抖动的数值参数：(名字, 下限, 上限, 抖动幅度)
KNOBS = (
    ("dentate_factor", 1, 4, 1),
    ("region_links", 0, 128, 32),
    ("n_time", 64, 1024, 256),
    ("k_near", 4, 24, 2),
    ("threshold_q", 60, 90, 5),        # 千分数：70 = 0.70（实测影响噪声鲁棒性）
)


def make_spec(rng):
    kw = {}
    for name, lo, hi, _ in KNOBS:
        kw[name] = int(rng.integers(lo, hi + 1))
    return kw


def mutate(kw, rng):
    out = dict(kw)
    for name, lo, hi, step in KNOBS:
        v = int(out.get(name, (lo + hi) // 2))
        v = int(np.clip(v + int(rng.integers(-step, step + 1)), lo, hi))
        out[name] = v
    return out


def score(kw, seed=0):
    """固定题库（快速子集）：感知三题 + 噪声 + 注视任务。"""
    kw = dict(kw)
    if "threshold_q" in kw:                 # 千分数 → 比例
        kw["threshold_q"] = float(kw["threshold_q"]) / 100.0
    try:
        sp = build_spec("full", seed=7 + seed, **kw)
        b = build(sp, verbose=False)
    except Exception:
        return 0.0
    ctx = Cortex(b)
    px = sp.px
    eng = b.engine

    def run(img, n=12):
        o = Observation(vision=np.asarray(img, np.float32),
                        audio=np.zeros((sp.audio_bands, sp.audio_frames), np.float32),
                        touch=np.full(sp.touch_ch, 0.5, np.float32),
                        proprio=np.full(sp.proprio_ch, 0.5, np.float32),
                        balance=np.full(sp.balance_ch, 0.5, np.float32))
        eng.act[:] = 0.0
        eng.reset_adaptation()
        for _ in range(n):
            ctx.tick(o, record=False)
        return set(np.asarray(b.pfc_code()).tolist())

    def img(ox=0.5, oy=0.5, seed=None):
        if seed is None:
            base = np.full((px, px, 3), 0.05, np.float32)
            return np.maximum(base, _blob(px, ox, oy, 0.25, (0.95, 0.2, 0.1)))
        return np.random.default_rng(seed).random((px, px, 3), dtype=np.float32)

    def jac(a, c):
        return len(a & c) / max(1, len(a | c))

    hit, tot = 0, 0
    a = run(img())
    tot += 1; hit += (a == run(img()))
    tot += 1; hit += jac(a, run(img(0.54, 0.5))) > 0.6
    tot += 1; hit += jac(a, run(img(seed=9))) < 0.4
    # 噪声题
    n = np.clip(img() + np.random.default_rng(3).normal(0, 0.03, (px, px, 3)), 0, 1)
    tot += 1; hit += jac(a, run(n.astype(np.float32))) > 0.5
    # 注视任务（两 episode，用同一个脑）
    from .tasks import GazeWorld
    ok = 0
    for s in range(2):
        w = GazeWorld(seed=s); o = w.reset()
        while not w.done():
            off = o
            x = 0.5 + 0.45 * float(np.clip(off[0], -1, 1))
            y = 0.5 + 0.45 * float(np.clip(off[1], -1, 1))
            base = np.full((px, px, 3), 0.05, np.float32)
            im = np.maximum(base, _blob(px, x, y, 0.18, (0.95, 0.2, 0.1)))
            oo = Observation(vision=im,
                             audio=np.zeros((sp.audio_bands, sp.audio_frames), np.float32),
                             touch=np.full(sp.touch_ch, 0.5, np.float32),
                             proprio=np.full(sp.proprio_ch, 0.5, np.float32),
                             balance=np.full(sp.balance_ch, 0.5, np.float32))
            ctx.tick(oo, record=False)
            f = np.asarray(b.forces(), np.float32)
            o = w.step(np.clip(np.array([f[2] - f[3], f[9] - f[10]], np.float32) * 2, -1, 1))
        ok += (abs(float(w.ball[0] - w.gaze[0])) < 0.12
               and abs(float(w.ball[1] - w.gaze[1])) < 0.12)
    tot += 1; hit += ok / 2.0
    return hit / tot


def evolve(gens=2, pop=3, seed=0, path=None):
    rng = np.random.default_rng(2024 + seed)
    best = None
    for g in range(gens):
        rows = []
        if best is None:
            cands = [make_spec(rng) for _ in range(pop)]
        else:
            # 精英保留 + 融合 + 变异（老库的"融合"= 参数按位平均）
            rows.append((best[0], best[1]))
            kids = [fuse(best[1], mutate(best[1], rng))]
            kids += [mutate(best[1], rng) for _ in range(max(0, pop - 2))]
            cands = kids
        for kw in cands:
            s = score(kw, seed)
            rows.append((s, kw))
            print(f"  第{g+1}代 {kw} → {s:.2f}", flush=True)
        rows.sort(key=lambda r: -r[0])
        if best is None or rows[0][0] > best[0]:
            best = rows[0]
        print(f"第 {g+1} 代最优 {best[0]:.2f} {best[1]}", flush=True)
    out = {"score": best[0], **best[1]}
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def fuse(a, b):
    """按位平均两个参数组（老库的"融合"就是参数平均）。"""
    out = {}
    for name, lo, hi, _ in KNOBS:
        va = int(a.get(name, (lo + hi) // 2))
        vb = int(b.get(name, (lo + hi) // 2))
        out[name] = int(round((va + vb) / 2.0))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--pop", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    best = evolve(a.gens, a.pop, a.seed, os.path.join(HERE, "evolve_best.json"))
    print("最优:", best)
