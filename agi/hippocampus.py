# -*- coding: utf-8 -*-
"""海马体：一个时间环。

每拍恰好一个时间神经元亮，位置顺序推进、走满一圈回到开头。这样"什么时候"
就有了表示——同样的特征在不同时刻亮，会和不同的时间神经元一起亮。

用法（由 Cortex 每拍调用一次）：
    ring = TimeRing(n=512)
    ring.step(engine)          # 把上一拍亮的时间神经元灭掉、下一个点亮

记忆的挂法（下一步）：特征神经元 ↔ 当时亮着的那个时间神经元，双向长连接。
"""
from __future__ import annotations

import numpy as np


class TimeRing:
    """固定长度的时间环，每拍恰好一个位置亮。"""

    def __init__(self, ids: np.ndarray, ticks_per_slot: int = 1):
        self.ids = np.asarray(ids, np.int64)
        self.n = len(self.ids)
        self.ticks_per_slot = max(1, int(ticks_per_slot))
        self.t = 0

    # ---------------------------------------------------------- 走
    def step(self, engine) -> int:
        """推进一拍，返回这一拍亮的时间神经元（全局编号）。"""
        idx = (self.t // self.ticks_per_slot) % self.n
        engine.set_external(self.ids[idx:idx + 1], np.ones(1, np.float32))
        self.t += 1
        return int(self.ids[idx])

    def current(self) -> int:
        return int(self.ids[(self.t // self.ticks_per_slot) % self.n])

    def reset(self) -> None:
        self.t = 0

    def __len__(self) -> int:
        return self.n
