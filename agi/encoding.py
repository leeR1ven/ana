# -*- coding: utf-8 -*-
"""感觉编码：把外界数值变成"每对恰好亮一个"的温度计细胞。

规则（来自链式架构）：
  · 每个通道 N 档 × 正反一对，**每对恰好亮一个**；
  · 纯黑时反细胞全亮——"什么都没有"也是一个读数，不是静默；
  · 所以输入层永远恰好一半活跃（每通道 N 个），不会有的图很亮、有的图很暗。

为什么不用二值阈值：温度计保留"差多少"的序信息；下一层的就近连线才能
从"亮到第几档"里读出强弱，而不是只知道"过线了"。

`time` 区是例外：时刻环每拍恰好一个亮（one-hot），不是温度计。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    """一拍的感觉。所有值都在 [0,1]。"""

    vision: np.ndarray | None = None          # (px, px, 3)
    audio: np.ndarray | None = None           # (bands, frames)
    touch: np.ndarray | None = None           # (touch_ch,)
    proprio: np.ndarray | None = None         # (proprio_ch,)
    balance: np.ndarray | None = None         # (balance_ch,)
    tick: int = 0
    meta: dict = field(default_factory=dict)


def thermometer(vals: np.ndarray, n_level: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (pos, neg) 两个 bool 掩码，形状 (C, N)；每行恰好 N 个 True（pos+neg）。"""
    v = np.clip(np.asarray(vals, np.float32).ravel(), 0.0, 1.0)
    lv = np.clip(np.rint(v * n_level), 0, n_level).astype(np.int16)
    j = np.arange(n_level, dtype=np.int16)[None, :]
    pos = lv[:, None] > j          # 亮到第 lv 档
    return pos, ~pos               # neg 是补集：每行加起来恰好 N


class Encoder:
    """把 Observation 写进输入细胞缓冲。输入细胞是引擎最前面的一段连续编号。"""

    def __init__(self, spec, layout: dict[str, tuple[int, int]], n_in: int):
        self.spec = spec
        self.layout = dict(layout)
        self.n_in = int(n_in)
        self.buf = np.zeros(self.n_in, np.float32)
        self.last = None

    # -------------------------------------------------- 各模态 -> 温度计
    def _fill_thermo(self, region: str, vals: np.ndarray) -> None:
        a, b = self.layout[region]
        n = self.spec.n_level
        c = (b - a) // (2 * n)          # 布局长度是 2×通道数×档数
        v = np.asarray(vals, np.float32).ravel()
        if len(v) != c:
            raise ValueError(f"{region} 期望 {c} 个通道，拿到 {len(v)}")
        pos, neg = thermometer(v, n)
        # 输入细胞的排布：[通道 0 的 pos N 个][通道 0 的 neg N 个][通道 1 ...]
        buf = self.buf[a:b].reshape(c, 2, n)
        buf[:, 0, :] = pos
        buf[:, 1, :] = neg

    def encode(self, obs: Observation) -> np.ndarray:
        sp = self.spec
        self.buf[:] = 0.0
        if "vision" in self.layout:
            img = obs.vision
            if img is None:
                img = np.zeros((sp.px, sp.px, 3), np.float32)
            self._fill_thermo("vision", np.asarray(img, np.float32).reshape(-1))
        if "audio" in self.layout:
            au = obs.audio if obs.audio is not None else np.zeros(
                (sp.audio_bands, sp.audio_frames), np.float32)
            self._fill_thermo("audio", np.asarray(au, np.float32).reshape(-1))
        for r in ("touch", "proprio", "balance"):
            if r not in self.layout:
                continue
            n = {"touch": sp.touch_ch, "proprio": sp.proprio_ch,
                 "balance": sp.balance_ch}[r]
            v = getattr(obs, r)
            self._fill_thermo(r, np.zeros(n, np.float32) if v is None else v)
        self.last = obs
        return self.buf

    def input_slice(self, region: str) -> slice:
        a, b = self.layout[region]
        return slice(a, b)
