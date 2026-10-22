# -*- coding: utf-8 -*-
"""批量打分：一个小模型 × 一整套小测，用来从上千个候选里筛。

为什么要"小模型 + 很多小测"：
  · 上千个候选 → 每个都必须很快（用 small 预设，控制在 1~2 秒）；
  · 分辨率来自"小测的数量"，不是"每道题有多难"——几百个小测就能把模型排出高低。

每个小测只做一件事：某个区、某条通道/某一段，动一档，看变化有没有传到更深一层。
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .spec import build_spec
from .build import build
from .encoding import Observation

BASE, BUMP = 0.34, 0.67


def _obs(sp, vision, audio, touch, proprio, balance):
    return Observation(vision=vision, audio=audio, touch=touch,
                       proprio=proprio, balance=balance)


def _blank(sp, px):
    return (np.full((px, px, 3), BASE, np.float32),
            np.full(sp.audio_bands * sp.audio_frames, BASE, np.float32),
            np.full(sp.touch_ch, BASE, np.float32),
            np.full(sp.proprio_ch, BASE, np.float32),
            np.full(sp.balance_ch, BASE, np.float32))


def _run(eng, enc, obs, n=8):
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(n):
        eng.clear_external()
        eng.set_input(enc.encode(obs))
        eng.step()
    return np.asarray(eng.act).copy()


def battery(sp, k_vision=64, k_other=24, n=8):
    """生成一批小测：返回 (名称, 观察) 的列表，成对（基线 + 改动）。"""
    px = sp.px
    v, a, t, p, b = _blank(sp, px)
    cases = [("base", _obs(sp, v.copy(), a.copy(), t.copy(), p.copy(), b.copy()))]
    flat = px * px * 3
    for i in range(k_vision):
        vv = v.reshape(-1).copy()
        s = (i * max(1, flat // k_vision)) % flat
        vv[s:s + max(1, flat // 10)] = BUMP
        cases.append((f"vision{i}", _obs(sp, vv.reshape(px, px, 3),
                                         a.copy(), t.copy(), p.copy(), b.copy())))
    for key, n_ch in (("audio", sp.audio_bands * sp.audio_frames),
                      ("touch", sp.touch_ch), ("proprio", sp.proprio_ch),
                      ("balance", sp.balance_ch)):
        for i in range(min(k_other, n_ch)):
            vec = {"audio": a, "touch": t, "proprio": p, "balance": b}[key].copy()
            k = max(1, n_ch // 8)
            vec[(i * k) % n_ch: (i * k) % n_ch + k] = BUMP
            args = dict(audio=a.copy(), touch=t.copy(), proprio=p.copy(), balance=b.copy())
            args[key] = vec
            cases.append((f"{key}{i}", _obs(sp, v.copy(), args["audio"],
                                            args["touch"], args["proprio"],
                                            args["balance"])))
    return cases


def score_spec(kw, preset="small", seed=0):
    """返回 (通过的小测数, 总数, 用时秒)。"""
    t0 = time.perf_counter()
    try:
        sp = build_spec(preset, seed=7 + seed, **kw)
        b = build(sp, verbose=False)
    except Exception:
        return 0, 0, time.perf_counter() - t0
    eng, enc = b.engine, b.encoder
    cases = battery(sp)
    acts = {name: _run(eng, enc, obs) for name, obs in cases}
    base = acts["base"]
    passed, total = 0, 0
    for name, _ in cases[1:]:
        a1 = acts[name]
        region = name.rstrip("0123456789")
        # 每个小测同时看 L2 和 L4：两道独立的小测（"传得进"和"传得深"）
        for depth in (".L2", ".L4"):
            key = f"{region}{depth}"
            L = b.layers.get(key)
            total += 1
            if L is not None:
                if np.abs(a1[L.ids] - base[L.ids]).max() > 1e-4:
                    passed += 1
            elif np.abs(a1 - base).max() > 1e-4:
                passed += 1
    return passed, total, time.perf_counter() - t0


def load_best(path="sweep_best.json"):
    import json
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_best(kw, score, path="sweep_best.json"):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"score": score, **kw}, f, ensure_ascii=False, indent=2)


def refine(base, n=60, preset="small", step=2, seed=1):
    """以已有最优为种子，在其附近抖动，找更好的。"""
    rng = np.random.default_rng(seed)
    best = dict(base)
    p, t, _ = score_spec({k: best[k] for k in ("region_links", "n_time", "k_near")}, preset)
    best_score = p / max(1, t)
    print(f"起点 {p}/{t} = {best_score:.2f} {best}")
    ranges = {"region_links": (0, 128), "n_time": (64, 1024), "k_near": (4, 24)}
    for i in range(n):
        kw = dict(best)
        for k, (lo, hi) in ranges.items():
            kw[k] = int(np.clip(kw[k] + int(rng.integers(-step, step + 1)), lo, hi))
        p, t, dt = score_spec({k: kw[k] for k in ranges}, preset)
        s = p / max(1, t)
        if s > best_score:
            best, best_score = dict(kw), s
            print(f"  #{i+1} 新高 {p}/{t} = {s:.2f} {best}")
        elif (i + 1) % 10 == 0:
            print(f"  #{i+1} 当前最好 {best_score:.2f}")
    print(f"最好 {best_score:.2f} {best}")
    return best, best_score


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--refine", action="store_true")
    a = ap.parse_args(argv)
    if a.refine:
        base = load_best() or {"region_links": 5, "n_time": 136, "k_near": 4}
        best, s = refine(base, a.n, a.preset)
        save_best({k: best[k] for k in ("region_links", "n_time", "k_near")}, s)
        return 0
    rng = np.random.default_rng(0)
    rows = []
    for i in range(a.n):
        kw = dict(dentate_factor=int(rng.integers(1, 4)),
                  region_links=int(rng.integers(0, 129)),
                  n_time=int(rng.integers(64, 1025)),
                  k_near=int(rng.integers(4, 25)))
        if "small" == a.preset:
            kw.pop("dentate_factor", None)      # 小模型不建扩维层
        p, t, dt = score_spec(kw, a.preset)
        rows.append((p / max(1, t), p, t, dt, kw))
        print(f"  {i+1}: {p}/{t} = {p/max(1,t):.2f}  用时 {dt:.1f}s")
    rows.sort(key=lambda r: -r[0])
    print("最好:", rows[0][1], "/", rows[0][2], rows[0][4])
    print("平均用时 %.1fs" % (sum(r[3] for r in rows) / len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
