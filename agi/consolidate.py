# -*- coding: utf-8 -*-
"""短期 → 长期：反复出现的"先 A 后 B"，才写进皮层连接。

海马体的绑定是"一次就写、很快忘"；皮层连接相反——要**反复出现**才长，
长起来就难改。这个模块负责搭桥：

    · 每拍记下"上一拍的码 → 这一拍的码"这个转移；
    · 同一个转移出现第 N 次（默认 3 次）时，才在两组单元之间写长期边；
    · 写的时候按"凹增长"，同一转移反复出现会越来越强，但有上限。

边组默认 65，每 `flush_every` 拍写进引擎。
"""
from __future__ import annotations

import numpy as np


class Consolidator:
    def __init__(self, engine, group: int = 65, repeats: int = 3,
                 lr: float = 0.05, cap: int = 64, flush_every: int = 50,
                 decay: float = 0.999, decay_every: int = 200, min_w: float = 0.02):
        self.eng = engine
        self.group = int(group)
        self.repeats = max(2, int(repeats))
        self.lr = float(lr)
        self.cap = int(cap)
        self.flush_every = max(1, int(flush_every))
        self.decay, self.decay_every, self.min_w = float(decay), max(1, int(decay_every)), float(min_w)
        self.count: dict[int, int] = {}       # 转移签名 -> 出现次数
        self.first: dict[int, np.ndarray] = {}
        self.w: dict[int, float] = {}
        self.prev: np.ndarray | None = None
        self.t = 0

    # ---------------------------------------------------------- 每拍
    @staticmethod
    def _sig(prev: np.ndarray, cur: np.ndarray) -> int:
        """转移签名：只取两端的"身份"，不看顺序细节（够用即可）。"""
        # 用"编号和"做签名：只取第一个单元太敏感（每次跑第一个可能不同），
        # 一整组的和稳定得多，又能区分不同内容
        a = int(prev.sum() % 1_000_003) if len(prev) else -1
        b = int(cur.sum() % 1_000_003) if len(cur) else -1
        return a * 1_000_003 + b * 31 + len(cur)

    def step(self, active) -> None:
        cur = np.asarray(sorted(set(int(x) for x in active)), np.int64)
        if self.prev is not None and len(self.prev) and len(cur):
            key = self._sig(self.prev, cur)
            c = self.count.get(key, 0) + 1
            self.count[key] = c
            if c >= self.repeats:          # 之后每次出现都再加强一点
                # 模式分离：只连"各自独有"的部分。重叠的单元本来就已经相互关联，
                # 再给它们加边只会把两段经历并成一个吸引子（实测 B 的独有单元掉到 8 个）。
                a = np.setdiff1d(self.prev, cur)[: self.cap]
                b = np.setdiff1d(cur, self.prev)[: self.cap]
                for s in a.tolist():
                    for d in b.tolist():
                        if s == d:
                            continue
                        k = s * self.eng.n + d
                        v = self.w.get(k, 0.0)
                        self.w[k] = min(1.0, v + self.lr * (1.0 - v))
        self.prev = cur
        self.t += 1
        if self.t % self.decay_every == 0:      # 久不用的长期边慢慢变弱
            f = self.decay ** self.decay_every
            for k in list(self.w.keys()):
                v = self.w[k] * f
                if v < self.min_w:
                    del self.w[k]
                else:
                    self.w[k] = v

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
    """回忆读出：不参与名额竞争，只看"驱动明显高于背景"的单元。

    为什么单独一条：扩维层每拍只有 2% 的名额，点 A 时名额被 A 自己占满，
    被联想起来的 B 即使驱动是门槛的 200 倍也进不了"亮的名单"。
    真实的回忆本来就该是另一条读出路，不该跟当前感知抢名额。
    """
    d = np.asarray(engine.drive())[np.asarray(ids)]
    base = float(np.median(d)) + 1e-9
    thr = max(float(floor), base * float(margin))
    ids = np.asarray(ids)
    return set(ids[d > thr].tolist())