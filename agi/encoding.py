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
    def _fovea_gain(self, c: int) -> np.ndarray:
        """中心高分辨：画面中心像素的增益高、边缘低（像视网膜的中央凹）。

        c 是通道数（px*px*3），按像素半径给一个平滑衰减的增益。
        """
        sp = self.spec
        if c % 3:
            return None
        npx = c // 3
        px = int(round(npx ** 0.5))
        if px * px != npx:
            return None
        yy, xx = np.mgrid[0:px, 0:px].astype(np.float32)
        cy = cx = (px - 1) / 2.0
        r = np.sqrt(((yy - cy) / max(1e-6, cy)) ** 2 + ((xx - cx) / max(1e-6, cx)) ** 2) / 1.4142
        g = 0.35 + 0.65 * np.exp(-(r ** 2) / (2 * 0.45 ** 2))   # 中心≈1.0，边缘≈0.35
        g = np.repeat(g[:, :, None], 3, axis=2).reshape(-1)
        if g.size != c:
            return None
        return g

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


    def _foveal(self, img: np.ndarray) -> np.ndarray:
        """中央凹重采样：中心像素密度高（一个小斑在中心会被放大），边缘稀疏。

        做法是"反向映射"：给 px×px 个采样点分配方向/半径，
        半径用指数映射 r = expm1(k·u)/expm1(k)，u 均匀 → 中心分到更多采样点。
        """
        sp = self.spec
        px = int(sp.px)
        kv = float(getattr(sp, "fovea_k", 2.2))
        u = np.linspace(-1.0, 1.0, px, dtype=np.float32)
        uu, vv = np.meshgrid(u, u)
        ru = np.maximum(np.abs(uu), np.abs(vv))          # 切比雪夫半径 ∈ [0,1]
        k = kv
        rr = np.expm1(k * ru) / np.expm1(k)              # 中心密集
        scale = np.where(ru > 1e-6, rr / np.maximum(ru, 1e-6), 1.0)
        sx = np.clip(((uu * scale + 1.0) * 0.5 * (px - 1)), 0, px - 1).astype(np.int32)
        sy = np.clip(((vv * scale + 1.0) * 0.5 * (px - 1)), 0, px - 1).astype(np.int32)
        return img[sy, sx, :]

    def encode(self, obs: Observation) -> np.ndarray:
        sp = self.spec
        self.buf[:] = 0.0
        if "vision" in self.layout:
            img = obs.vision
            if img is None:
                img = np.zeros((sp.px, sp.px, 3), np.float32)
            if getattr(sp, "fovea", True):
                img = self._foveal(np.asarray(img, np.float32))
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
