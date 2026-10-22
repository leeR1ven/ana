# -*- coding: utf-8 -*-
"""区与区之间的"短路"：相邻区同时有活动时，从活跃单元长一条直连（绕开前额叶）。

为什么单独一个模块：它只关心"哪些区的单元同时亮着"，与学习规则、PFC 拓扑无关。
独立出来可以默认启用，出问题也容易单独关掉（`Spec.region_shortcut = False`）。

规则（凹增长，权重有上限）：
    区 r 的活跃单元  →  区 r+1 的活跃单元       （两个方向都长）
写得快（每个活跃对一次就成长），每 `flush_every` 拍写进引擎（组 58）。
"""
from __future__ import annotations

import numpy as np


class RegionShortcuts:
    def __init__(self, engine, group: int = 58, lr: float = 0.02,
                 cap: int = 16, flush_every: int = 50, seed: int = 11):
        self.eng = engine
        self.group = int(group)
        self.lr = float(lr)
        self.cap = int(cap)
        self.flush_every = max(1, int(flush_every))
        self.rng = np.random.default_rng(seed)
        self.w: dict[int, float] = {}
        self.t = 0

    def step(self, active) -> None:
        """每拍调用：把相邻区里同时活跃的单元连起来。"""
        act = np.asarray(active, np.int64)
        if act.size < 2:
            return
        reg = self.eng.region[act]
        for r in np.unique(reg):
            a = act[reg == r]
            b = act[reg == (r + 1)]
            if not len(a) or not len(b):
                continue
            if len(a) > self.cap:
                a = a[self.rng.choice(len(a), self.cap, replace=False)]
            if len(b) > self.cap:
                b = b[self.rng.choice(len(b), self.cap, replace=False)]
            for s in a.tolist():
                for d in b.tolist():
                    if s == d:
                        continue
                    k = s * self.eng.n + d
                    v = self.w.get(k, 0.0)
                    self.w[k] = min(1.0, v + self.lr * (1.0 - v))
        self.t += 1

    def flush(self) -> int:
        if not self.w:
            return 0
        old = self.eng.group == self.group
        if old.any():
            self.eng.drop_edges(old)
        ks = np.fromiter(self.w.keys(), np.int64, len(self.w))
        vs = np.fromiter(self.w.values(), np.float32, len(self.w))
        self.eng.add_edges((ks // self.eng.n).astype(np.int32),
                           (ks % self.eng.n).astype(np.int32), vs,
                           group=self.group)
        self.eng.rebuild_backend()
        return len(ks)
