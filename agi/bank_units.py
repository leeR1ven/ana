# -*- coding: utf-8 -*-
"""小测（一档一测）：照老库 `验收标准.md` / `train_loop.evaluate` 的做法。

每个小测只做一件事：**某个区、某个通道，动最小一档**，然后看这个变化有没有传到
L2 / L4 / 前额叶。这样一次能得到几十个小测，每个都能独立定位问题。

用法：python -m agi.bank_units [--seed 0]
"""
from __future__ import annotations

import argparse

import numpy as np

from .bank import Harness, scene
from .encoding import Observation


BASE = 0.34
BUMP = 0.67


def _obs(h, vision=None, audio=None, touch=None, proprio=None, balance=None):
    sp = h.sp
    img = np.full((h.px, h.px, 3), BASE, np.float32) if vision is None else vision
    return Observation(
        vision=img,
        audio=np.full(sp.audio_bands * sp.audio_frames, BASE, np.float32)
        if audio is None else audio,
        touch=np.full(sp.touch_ch, BASE, np.float32) if touch is None else touch,
        proprio=np.full(sp.proprio_ch, BASE, np.float32) if proprio is None else proprio,
        balance=np.full(sp.balance_ch, BASE, np.float32) if balance is None else balance)


def _run(h, obs, n=40):
    h.eng.act[:] = 0.0
    h.eng.reset_adaptation()
    for _ in range(n):
        h.ctx.tick(obs, record=False)
    a = np.asarray(h.eng.act).copy()
    out = {}
    for name, L in h.b.layers.items():
        if name.endswith(".L2") or name.endswith(".L4"):
            out[name] = a[L.ids].copy()
    out["pfc"] = a[np.asarray(h.b.pfc_ids)].copy()
    return out


def measure(h, seed=0):
    rng = np.random.default_rng(seed)
    base = _run(h, _obs(h))
    rows = []

    # 视觉：抽几个像素通道
    flat = h.px * h.px * 3
    for c in rng.choice(flat, min(12, flat), replace=False):
        img = np.full((h.px, h.px, 3), BASE, np.float32).reshape(-1)
        # 一个小块（约 10% 通道）而不是单像素：实测单像素在 L2 就被稀释掉了
        k = max(1, flat // 10)
        img[c:c + k] = BUMP
        a1 = _run(h, _obs(h, vision=img.reshape(h.px, h.px, 3)))
        rows.append(_judge("vision", int(c), base, a1))

    # 其余区：逐通道
    sp = h.sp
    for key, n in (("audio", sp.audio_bands * sp.audio_frames),
                   ("touch", sp.touch_ch), ("proprio", sp.proprio_ch),
                   ("balance", sp.balance_ch)):
        for c in range(min(n, 12)):
            v = np.full(n, BASE, np.float32)
            k = max(1, n // 8)          # 按区规模换算：改一整段，不是一个通道
            v[c:c + k] = BUMP
            a1 = _run(h, _obs(h, **{key: v}))
            rows.append(_judge(key, c, base, a1))
    return rows


def _judge(region, ch, base, a1):
    """变化有没有传到：本区 L2 / L4 / 前额叶。"""
    out = {"region": region, "channel": ch}
    for tag, key in (("L2", f"{region}.L2"), ("L4", f"{region}.L4")):
        b0, b1 = base.get(key), a1.get(key)
        out[tag] = bool(b0 is not None and b1 is not None
                        and np.abs(b1 - b0).max() > 1e-4)
    out["pfc"] = bool(np.abs(a1["pfc"] - base["pfc"]).max() > 1e-4)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    h = Harness(a.seed)
    rows = measure(h, a.seed)
    from collections import Counter
    by = {}
    for r in rows:
        d = by.setdefault(r["region"], Counter())
        d["n"] += 1
        d["L2"] += r["L2"]
        d["L4"] += r["L4"]
        d["pfc"] += r["pfc"]
    print("逐通道小测（动一档，看有没有传到）")
    for region, c in by.items():
        n = c["n"]
        print("  %-8s n=%2d  L2 %2d/%2d  L4 %2d/%2d  前额叶 %2d/%2d"
              % (region, n, c["L2"], n, c["L4"], n, c["pfc"], n))
    tot = Counter()
    for c in by.values():
        for k in ("n", "L2", "L4", "pfc"):
            tot[k] += c[k]
    print("  合计     n=%d  L2 %d/%d  L4 %d/%d  前额叶 %d/%d"
          % (tot["n"], tot["L2"], tot["n"], tot["L4"], tot["n"], tot["pfc"], tot["n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
