# -*- coding: utf-8 -*-
"""短期 → 长期：反复出现的"先 A 后 B"，才写进皮层连接。

海马体的绑定是"一次就写、很快忘"；皮层连接相反——要**反复出现**才长，
长起来就难改，但久不用也会慢慢变弱（遗忘曲线）。

关键细节：**"同一个经历"不能要求一字不差**。两次看同一张图，扩维层的码可能差几个单元，
所以这里用"两端的交集比例都超过 0.8"来判定是不是同一种转移。
"""
from __future__ import annotations

import numpy as np


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / float(min(len(a), len(b)))


class Consolidator:
    def __init__(self, engine, group: int = 65, repeats: int = 3,
                 lr: float = 0.05, cap: int = 64, flush_every: int = 50,
                 match: float = 0.8):
        self.eng = engine
        self.group = int(group)
        self.repeats = max(2, int(repeats))
        self.lr = float(lr)
        self.cap = int(cap)
        self.flush_every = max(1, int(flush_every))
        self.match = float(match)
        self.templates: list[dict] = []       # 见过的转移（模糊匹配）
        self.w: dict[int, float] = {}
        self.prev: np.ndarray | None = None
        self.t = 0

    # ---------------------------------------------------------- 每拍
    def _find(self, cur: np.ndarray):
        for tp in self.templates:
            if (_overlap(self.prev, tp["a"]) > self.match
                    and _overlap(cur, tp["b"]) > self.match):
                return tp
        return None

    def step(self, active) -> None:
        cur = np.asarray(sorted(set(int(x) for x in active)), np.int64)
        if self.prev is not None and len(self.prev) and len(cur):
            tp = self._find(cur)
            if tp is None:
                tp = dict(a=self.prev, b=cur, n=1)
                self.templates.append(tp)
            else:
                tp["n"] += 1
            if tp["n"] >= self.repeats:      # 反复出现的，每次再加强一点
                a = np.asarray(tp["a"])[: self.cap]
                b = np.asarray(tp["b"])[: self.cap]
                for s in a.tolist():
                    for d in b.tolist():
                        if s == d:
                            continue
                        k = s * self.eng.n + d
                        v = self.w.get(k, 0.0)
                        self.w[k] = min(1.0, v + self.lr * (1.0 - v))
        self.prev = cur
        self.t += 1

    # ---------------------------------------------------------- 写入
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


def recall_readout(engine, ids, margin: float = 8.0, floor: float = 0.5):
    """回忆读出：不参与名额竞争，只看"驱动明显高于背景"的单元。"""
    ids = np.asarray(ids)
    d = np.asarray(engine.drive())[ids]
    base = float(np.median(d)) + 1e-9
    thr = max(float(floor), base * float(margin))
    return set(ids[d > thr].tolist())