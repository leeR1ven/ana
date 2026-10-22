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


class HippoBinder:
    """特征 ↔ 时间 的绑定：某一刻亮着的前额叶单元，和那一刻的时间神经元互相连。

    这正是 born-wired-cortex 的做法（视觉/听觉/运动记忆都是"特征↔时间"双向连接）。
    权重按共现次数凹增长，用得多的变难改（固化）；每 flush_every 拍写进引擎。
    """

    def __init__(self, engine, pfc_ids, ring, group_in: int = 63, group_out: int = 64,
                 lr: float = 1.0, cap: int = 256, w_max: float = 1.0,
                 decay: float = 0.99, decay_every: int = 10, min_w: float = 0.05,
                 strength: float = 3.0, seed: int = 5):
        # 时间绑定是**短期记忆**：写得快（lr≈1，一次就到目标）、忘得快（decay + 删弱边）。
        # 这和皮层/PFC 的长期连接相反（那边是凹增长、难改、不衰减）。
        self.eng = engine
        self.pfc = np.asarray(pfc_ids, np.int64)
        self.ring = ring
        self.g_in, self.g_out = group_in, group_out
        self.lr, self.cap, self.w_max = lr, int(cap), w_max
        self.decay, self.decay_every, self.min_w = decay, max(1, int(decay_every)), min_w
        self.strength = strength      # 时间→内容的边要够强才推得亮（感觉层门槛较高）
        self.rng = np.random.default_rng(seed)
        self.enc: dict[int, float] = {}     # pfc -> time 的权重
        self.dec: dict[int, float] = {}     # time -> pfc 的权重
        self.t = 0

    def _grow(self, d: dict, k: int) -> None:
        w = d.get(k, 0.0)
        d[k] = w + self.lr * max(0.0, 1.0 - w / self.w_max)

    def step(self, active_pfc) -> None:
        """每拍调用：把当前前额叶活动绑到当前时间格上。"""
        act = np.asarray(active_pfc, np.int64)
        if len(act) == 0:
            return
        if len(act) > self.cap:
            act = act[self.rng.choice(len(act), self.cap, replace=False)]
        t = (self.ring.t // self.ring.ticks_per_slot) % self.ring.n   # 本地格号（不是全局编号）
        for p in act.tolist():
            self._grow(self.enc, p * self.ring.n + t)     # 特征 → 时间
            self._grow(self.dec, t * 10_000_000 + p)          # 时间 → 特征
        self.t += 1
        if self.t % self.decay_every == 0:      # 短期记忆：定时衰减，弱边直接丢
            f = self.decay ** self.decay_every
            for d in (self.enc, self.dec):
                for k in list(d.keys()):
                    v = d[k] * f
                    if v < self.min_w:
                        del d[k]
                    else:
                        d[k] = v

    def flush(self) -> int:
        """把攒下的连接写进引擎（先清掉旧的，再整体重写）。

        只写"时间→内容"这一向：反方向会让前额叶的码点亮一串时间格、再反推回来，
        造成大面积扩散（实测码从 1,223 涨到 10,190）。"内容→时间"存字典即可。
        """
        if not self.dec:
            return 0
        old = self.eng.group == self.g_out
        if old.any():
            self.eng.drop_edges(old)
        keys = np.fromiter(self.dec.keys(), np.int64, len(self.dec))
        vals = np.fromiter(self.dec.values(), np.float32, len(self.dec))
        src = self.ring.ids[(keys // 10_000_000).astype(np.int64)]
        dst = (keys % 10_000_000).astype(np.int32)
        self.eng.add_edges(src, dst, (vals * self.strength).astype(np.float32), group=self.g_out)
        self.eng.rebuild_backend()
        return len(keys)