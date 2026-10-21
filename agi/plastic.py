# -*- coding: utf-8 -*-
"""联想连接（记忆）的生长规则——v0 默认关闭，边数 0。

这一层不是"外挂记忆库"，而是**前额叶神经元之间的连接**。前额叶的每个神经元
就像一个字/一个音：单独亮没有意义，靠"谁先亮、谁后亮"长出有向连接，才有关联、
才有回忆、才谈得上意识。所以：

  · 没有海马体，没有单独的 memory 模块；
  · v0 前额叶内部边 = 0（只有出生接线）；
  · 训练/测试阶段喂真实序列，调用 `PlasticRule.apply()` 让连接长出来。

**规则是"有先后的赫布"，不是"同时活跃就加强"**（这一条被纠正过）：

    上一拍亮 A、这一拍亮 B   →   A→B 加强        （时间顺序决定方向）
    这一拍刚亮的 B、C 之间   →   互相加强（弱）   （同一时刻的共同出现）
    一直亮 / 一直灭的不产生边                    （不携带顺序信息）

其余规则来自字神经元网络的实测（见 docs/MECHANISMS.md）：

1. **凹形生长**：Δw = η·a_i·a_j·(1 − w/w_max)^pow —— 越涨越难涨；
2. **稀疏**：每个单元只留最强的 top-k 条出边；
3. **整体降权**：每 N 拍 ×decay，让"从不用"的连接消失；
4. **连接强度 ≠ 传播强度**：Ŵ = W^1.5 / 出边和 / max(入边和, 下限)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlasticConfig:
    enabled: bool = False
    group: int = 40                 # 前额叶内部联想边（不要和其它组号复用）
    eta: float = 0.02               # 顺序赫布：先亮 → 后亮 的增量
    eta_sim: float = 0.008          # 同拍刚亮：互相加强的增量（弱一些）
    w_max: float = 1.0
    pow: float = 1.0                # (1 - w/w_max)^pow，越大越凹
    topk_out: int = 8               # 每个单元保留几条最强出边
    min_w: float = 0.01
    decay_every: int = 2000         # 每多少拍整体降权一次
    decay: float = 0.995
    max_pairs: int = 4096           # 一拍最多处理几对（防止爆）
    sharpen: float = 1.5            # 传播权重 = W^sharpen
    in_floor: float = 1.0           # 入边和下限（防除零）


class PlasticRule:
    """前额叶内的顺序赫布生长。v0 里 `enabled=False`，`apply()` 是空操作。"""

    def __init__(self, cfg: PlasticConfig | None = None, seed: int = 3):
        self.cfg = cfg or PlasticConfig()
        self.rng = np.random.default_rng(seed)
        self._idx: dict[int, int] = {}     # key = src*n+dst → edge index
        self._prev = np.zeros(0, np.int64)
        self._grown = 0
        self._strengthened = 0
        self.t = 0

    # ------------------------------------------------------------ 主入口
    def apply(self, engine, cur_active, prev_active=None) -> dict:
        """一拍结束调用。

        cur_active  : 这一拍活跃的前额叶编号
        prev_active : 上一拍活跃的编号（不给就用内部记住的上一拍）
        """
        if not self.cfg.enabled:
            return dict(edges=0, grown=0, strengthened=0, pruned=0)
        cur = np.unique(np.asarray(cur_active, np.int64))
        prev = self._prev if prev_active is None else np.unique(
            np.asarray(prev_active, np.int64))
        grown = strengthened = 0
        if len(prev) and len(cur):
            grown, strengthened = self._hebb(engine, prev, cur)
        pruned = 0
        self.t += 1
        if self.cfg.decay_every and self.t % self.cfg.decay_every == 0:
            self._decay(engine)
            pruned = self._prune(engine)
        if grown or strengthened or pruned:
            self._prune_out(engine)
            self.sync(engine)
            engine.rebuild_backend()
        self._prev = cur
        return dict(edges=self.count(engine), grown=grown,
                    strengthened=strengthened, pruned=pruned)

    # ------------------------------------------------------------ 赫布
    def _hebb(self, engine, prev: np.ndarray, cur: np.ndarray) -> tuple[int, int]:
        cfg = self.cfg
        self._grown = self._strengthened = 0
        n = engine.n
        mask_prev = np.zeros(n, bool)
        mask_prev[prev] = True
        newly = cur[~mask_prev[cur]]        # 这一拍"刚亮"的
        # 1) 顺序：上一拍亮 → 这一拍亮
        n_pair = min(cfg.max_pairs, len(prev) * len(cur))
        if n_pair > 0:
            ii = prev[self.rng.integers(0, len(prev), n_pair)]
            jj = cur[self.rng.integers(0, len(cur), n_pair)]
            keep = ii != jj
            self._apply_pairs(engine, ii[keep], jj[keep], cfg.eta)
        # 2) 同拍刚亮：互相加强（双向，弱）
        if len(newly) > 1:
            k = min(cfg.max_pairs // 2, len(newly) * max(1, len(newly) - 1) // 2)
            if k > 0:
                a = newly[self.rng.integers(0, len(newly), k)]
                b = newly[self.rng.integers(0, len(newly), k)]
                keep = a != b
                self._apply_pairs(engine, a[keep], b[keep], cfg.eta_sim)
        return self._grown, self._strengthened

    def _apply_pairs(self, engine, src: np.ndarray, dst: np.ndarray, eta: float) -> None:
        if not len(src):
            return
        n = engine.n
        keys = np.unique(src.astype(np.int64) * n + dst.astype(np.int64))
        s = (keys // n).astype(np.int64)
        d = (keys % n).astype(np.int64)
        new_s, new_d, new_w = [], [], []
        old_i, old_inc = [], []
        idx = self._idx
        for a, b in zip(s.tolist(), d.tolist()):
            i = idx.get(a * n + b)
            if i is None:
                new_s.append(a)
                new_d.append(b)
                new_w.append(eta)
            else:
                w = engine.w[i]
                inc = eta * max(0.0, 1.0 - w / self.cfg.w_max) ** self.cfg.pow
                if inc > 0:
                    old_i.append(i)
                    old_inc.append(inc)
        if old_i:
            engine.w[old_i] += np.asarray(old_inc, np.float32)
        if new_s:
            base = len(engine.w)
            engine.add_edges(np.asarray(new_s, np.int32), np.asarray(new_d, np.int32),
                             np.asarray(new_w, np.float32), group=self.cfg.group)
            for j, (a, b) in enumerate(zip(new_s, new_d)):
                idx[a * n + b] = base + j
        self._grown += len(new_s)
        self._strengthened += len(old_i)

    # ------------------------------------------------------------ 剪枝/降权
    def _prune_out(self, engine) -> None:
        cfg = self.cfg
        idx = np.flatnonzero(engine.group == cfg.group)
        if not len(idx):
            return
        src = engine.src[idx]
        w = engine.w[idx]
        order = np.lexsort((-w, src))
        idx_s, src_s = idx[order], src[order]
        starts = np.r_[0, np.flatnonzero(np.diff(src_s)) + 1]
        counts = np.diff(np.r_[starts, len(src_s)])
        rank = np.arange(len(src_s)) - np.repeat(starts, counts)
        drop = idx_s[rank >= cfg.topk_out]
        if len(drop):
            engine.drop_edges(np.isin(np.arange(len(engine.src)), drop))
            self._reindex(engine)

    def _prune(self, engine) -> int:
        idx = np.flatnonzero(engine.group == self.cfg.group)
        if not len(idx):
            return 0
        weak = idx[engine.w[idx] < self.cfg.min_w]
        if len(weak):
            engine.drop_edges(np.isin(np.arange(len(engine.src)), weak))
            self._reindex(engine)
        return int(len(weak))

    def _decay(self, engine) -> None:
        m = engine.group == self.cfg.group
        if m.any():
            engine.w[m] *= self.cfg.decay

    def _reindex(self, engine) -> None:
        idx = np.flatnonzero(engine.group == self.cfg.group)
        n = engine.n
        self._idx = {int(engine.src[i]) * n + int(engine.dst[i]): int(i) for i in idx}

    # ------------------------------------------------------------ 传播权重
    def sync(self, engine) -> None:
        """连接强度 W → 传播权重 Ŵ（扇出/扇入归一 + 锐化）。"""
        cfg = self.cfg
        m = engine.group == cfg.group
        if not m.any():
            return
        idx = np.flatnonzero(m)
        w = np.maximum(engine.w[idx], 0.0)
        out = np.bincount(engine.src[idx], weights=w, minlength=engine.n)
        inn = np.bincount(engine.dst[idx], weights=w, minlength=engine.n)
        ws = w ** cfg.sharpen
        denom = np.maximum(out[engine.src[idx]], 1e-6)
        denom2 = np.maximum(inn[engine.dst[idx]], cfg.in_floor)
        engine.wp[idx] = (ws / denom / denom2).astype(np.float32)

    def count(self, engine) -> int:
        return int((engine.group == self.cfg.group).sum())

    def stats(self, engine) -> dict:
        m = engine.group == self.cfg.group
        return dict(edges=int(m.sum()), enabled=self.cfg.enabled,
                    mean_w=float(engine.w[m].mean()) if m.any() else 0.0,
                    topk_out=self.cfg.topk_out)
