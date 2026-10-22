# -*- coding: utf-8 -*-
"""题库（第二册）：用"表格"批量定义题目，覆盖感知/抗干扰/记忆的各种变化。

第一册（agi/bank.py）是手写题；这里用声明式表格补足数量。
用法：python -m agi.bank_more
"""
from __future__ import annotations

import numpy as np

from .bank import Harness, case, jac, scene, run as run_bank
from .world import _blob


def _blob_img(h, ox, oy, r=0.25, color=(0.95, 0.2, 0.1), bg=0.05):
    base = np.full((h.px, h.px, 3), bg, np.float32)
    return np.maximum(base, _blob(h.px, ox, oy, r, color))


def _pair(name, cat, make_a, make_b, lo=None, hi=None):
    """一张对比题：两张图的码重合 j 要落在 [lo, hi] 区间。"""
    def fn(h):
        h.run(make_a(h))
        a = h.pfc()
        h.run(make_b(h))
        j = jac(a, h.pfc())
        if lo is not None and j < lo:
            return False
        if hi is not None and j > hi:
            return False
        return True
    case(cat, name)(fn)


# ---- 感知：同一物体，各种小变化（应该"像"）----
_pair("平移2%", "感知", lambda h: _blob_img(h, 0.50, 0.50),
      lambda h: _blob_img(h, 0.52, 0.50), lo=0.6)
_pair("平移8%", "感知", lambda h: _blob_img(h, 0.50, 0.50),
      lambda h: _blob_img(h, 0.58, 0.50), lo=0.4)
_pair("半径+2%", "感知", lambda h: _blob_img(h, 0.5, 0.5, 0.25),
      lambda h: _blob_img(h, 0.5, 0.5, 0.27), lo=0.5)

# ---- 感知：不同物体（应该"不像"）----
_pair("换颜色", "感知", lambda h: _blob_img(h, 0.5, 0.5),
      lambda h: _blob_img(h, 0.5, 0.5, color=(0.1, 0.9, 0.2)), hi=0.4)
_pair("换位置", "感知", lambda h: _blob_img(h, 0.25, 0.25),
      lambda h: _blob_img(h, 0.75, 0.75), hi=0.7)
_pair("换背景", "感知", lambda h: _blob_img(h, 0.5, 0.5, bg=0.05),
      lambda h: _blob_img(h, 0.5, 0.5, bg=0.45), hi=0.6)
_pair("大球vs小球", "感知", lambda h: _blob_img(h, 0.5, 0.5, 0.30),
      lambda h: _blob_img(h, 0.5, 0.5, 0.10), hi=0.6)

# ---- 抗干扰（应该仍然"像"）----
_pair("噪声1%", "抗干扰", lambda h: _blob_img(h, 0.5, 0.5),
      lambda h: np.clip(_blob_img(h, 0.5, 0.5)
                        + np.random.default_rng(2).normal(0, 0.01, (h.px, h.px, 3)), 0, 1).astype(np.float32),
      lo=0.6)
_pair("噪声5%", "抗干扰", lambda h: _blob_img(h, 0.5, 0.5),
      lambda h: np.clip(_blob_img(h, 0.5, 0.5)
                        + np.random.default_rng(2).normal(0, 0.05, (h.px, h.px, 3)), 0, 1).astype(np.float32),
      lo=0.4)
_pair("遮挡25%", "抗干扰", lambda h: _blob_img(h, 0.4, 0.4),
      lambda h: _occlude(_blob_img(h, 0.4, 0.4), h, 0.25), lo=0.4)
_pair("遮挡50%", "抗干扰", lambda h: _blob_img(h, 0.4, 0.4),
      lambda h: _occlude(_blob_img(h, 0.4, 0.4), h, 0.5), lo=0.3)
_pair("亮度+5%", "抗干扰", lambda h: _blob_img(h, 0.5, 0.5),
      lambda h: np.clip(_blob_img(h, 0.5, 0.5) + 0.05, 0, 1).astype(np.float32), lo=0.5)


def _occlude(img, h, frac):
    out = img.copy()
    k = int(h.px * frac)
    out[:k] = 0.05
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run_bank(a.seed)
