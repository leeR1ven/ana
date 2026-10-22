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


def battery(sp, k_vision=24, k_other=12, n=8):
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
    deep = [k for k in b.layers if k.endswith(".L4")]
    for name, _ in cases[1:]:
        a1 = acts[name]
        region = name.rstrip("0123456789")
        tgt = [k for k in deep if k.startswith(region)]
        total += 1
        if tgt:
            L = b.layers[tgt[0]]
            if np.abs(a1[L.ids] - base[L.ids]).max() > 1e-4:
                passed += 1
        else:                       # 没有对应深层就只要任何变化
            if np.abs(a1 - base).max() > 1e-4:
                passed += 1
    return passed, total, time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--preset", default="small")
    a = ap.parse_args(argv)
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
