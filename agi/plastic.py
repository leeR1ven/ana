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
    full_cross: int = 20000         # 两批单元乘积不超过这个数就全配对学（不抽样）
    min_change: float = 0.3         # 只在"画面明显变了"时学顺序（重叠<70% 才算换场景）
    flush_every: int = 25           # 每多少拍把攒下的新边真正写进引擎（重建张量很贵）
    flush_edges: int = 4000         # 攒够这么多条也提前写一次
    sharpen: float = 1.5            # 传播权重 = W^sharpen
    prop_gain: float = 1.0          # 传播权重的总增益（混合驱动时要抬，见 sync 注释）
    in_floor: float = 1.0           # 出边和下限（防除零）


class PlasticRule:
    """前额叶内的顺序赫布生长。v0 里 `enabled=False`，`apply()` 是空操作。"""

    def __init__(self, cfg: PlasticConfig | None = None, seed: int = 3):
        self.cfg = cfg or PlasticConfig()
        self.rng = np.random.default_rng(seed)
        self._idx: dict[int, int] = {}     # key = src*n+dst → 已写入引擎的边下标
        self._pending: dict[int, int] = {}  # key → 待写入缓冲区下标
        self._ns: list[int] = []
        self._nd: list[int] = []
        self._nw: list[float] = []
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
        self.t += 1
        pruned = 0
        due = (self._ns and (len(self._ns) >= self.cfg.flush_edges
                             or self.t % max(1, self.cfg.flush_every) == 0))
        if due:
            pruned = self.flush(engine)
        self._prev = cur
        return dict(edges=self.count(engine), grown=grown,
                    strengthened=strengthened, pruned=pruned,
                    pending=len(self._ns))

    def reinforce(self, engine, src, dst, eta: float | None = None) -> int:
        """监督式：把给定的 (src→dst) 对按凹增长加强（蒸馏"状态→动作"用）。

        和顺序赫布共用同一套存储与剪枝，只是"谁和谁该连"由外部示范给出。
        """
        if not self.cfg.enabled:
            return 0
        src = np.unique(np.asarray(src, np.int64))
        dst = np.unique(np.asarray(dst, np.int64))
        if not len(src) or not len(dst):
            return 0
        # 全连接对：每个源 → 每个目标（超过上限就随机抽样）
        s = np.repeat(src, len(dst))
        d = np.tile(dst, len(src))
        cap = self.cfg.max_pairs * 4
        if len(s) > cap:
            pick = self.rng.choice(len(s), cap, replace=False)
            s, d = s[pick], d[pick]
        self._apply_pairs(engine, s, d, float(eta if eta is not None
                                              else self.cfg.eta))
        if len(self._ns) >= self.cfg.flush_edges:
            self.flush(engine)
        return len(src)

    def flush(self, engine) -> int:
        """把攒着的新边真正写进引擎（会重建后端张量，所以别每拍都做）。"""
        if not self._ns:
            return 0
        engine.add_edges(np.asarray(self._ns, np.int32),
                         np.asarray(self._nd, np.int32),
                         np.asarray(self._nw, np.float32), group=self.cfg.group)
        base = len(engine.w) - len(self._ns)
        for j, k in enumerate(self._pending):
            self._idx[k] = base + j
        self._ns, self._nd, self._nw, self._pending = [], [], [], {}
        self._prune_out(engine)
        pruned = 0
        if self.cfg.decay_every and self.t % self.cfg.decay_every == 0:
            self._decay(engine)
            pruned = self._prune(engine)
        self.sync(engine)
        engine.rebuild_backend()
        return pruned

    # ------------------------------------------------------------ 赫布
    def _hebb(self, engine, prev: np.ndarray, cur: np.ndarray) -> tuple[int, int]:
        cfg = self.cfg
        self._grown = self._strengthened = 0
        # 同一画面持续时不学：那不是"先 A 后 B"，只是"还停在 A"。
        # 不排除它，自我关联会淹没真正的转移（实测 7/8 的拍都是这种情况）。
        if len(cur):
            overlap = len(np.intersect1d(prev, cur, assume_unique=True)) / len(cur)
            if overlap > 1.0 - cfg.min_change:
                return 0, 0
        n = engine.n
        mask_prev = np.zeros(n, bool)
        mask_prev[prev] = True
        newly = cur[~mask_prev[cur]]        # 这一拍"刚亮"的
        # 1) 顺序：上一拍亮 → 这一拍亮
        n_all = len(prev) * len(cur)
        if n_all <= cfg.full_cross:
            # 全配对：换画面的那一刻，把"上一批 × 这一批"全部学一遍。
            # 抽样会让真转移一万次里也抽不到几次（上一轮的教训）。
            ii = np.repeat(prev, len(cur))
            jj = np.tile(cur, len(prev))
            keep = ii != jj
            self._apply_pairs(engine, ii[keep], jj[keep], cfg.eta)
        else:
            n_pair = min(cfg.max_pairs, n_all)
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
        """处理采样到的 (src→dst) 对：已写入的就地加强，新的先攒在缓冲区。"""
        if not len(src):
            return
        n = engine.n
        keys = np.unique(src.astype(np.int64) * n + dst.astype(np.int64))
        s = (keys // n).tolist()
        d = (keys % n).tolist()
        old_i, old_inc = [], []
        grown = 0
        for a, b in zip(s, d):
            k = a * n + b
            i = self._idx.get(k)
            if i is not None:
                w = engine.w[i]
                inc = eta * max(0.0, 1.0 - w / self.cfg.w_max) ** self.cfg.pow
                if inc > 0:
                    old_i.append(i)
                    old_inc.append(inc)
                continue
            j = self._pending.get(k)
            if j is not None:
                w = self._nw[j]
                inc = eta * max(0.0, 1.0 - w / self.cfg.w_max) ** self.cfg.pow
                self._nw[j] = min(self.cfg.w_max, w + inc)
                continue
            self._pending[k] = len(self._ns)
            self._ns.append(a)
            self._nd.append(b)
            self._nw.append(float(eta))
            grown += 1
        if old_i:
            engine.w[old_i] += np.asarray(old_inc, np.float32)
        self._grown += grown
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
        """连接强度 W → 传播权重 Ŵ。

        这里和字神经元网络有一处**有意的偏差**：原库除"出边和"之后再除"入边和"，
        因为那里的 W 是唯一驱动。我们这里前额叶还有一条 1:1 的感觉线，
        两份归一叠起来会把联想压到 1%（实测 0.0065/边，占驱动 1.17%），
        等于白学。所以只保留扇出归一（一个源把总输出摊到它的 top-k 上），
        再做 W^sharpen 锐化，最后乘 prop_gain。
        """
        cfg = self.cfg
        m = engine.group == cfg.group
        if not m.any():
            return
        idx = np.flatnonzero(m)
        w = np.maximum(engine.w[idx], 0.0)
        out = np.bincount(engine.src[idx], weights=w, minlength=engine.n)
        ws = w ** cfg.sharpen
        denom = np.maximum(out[engine.src[idx]], cfg.in_floor)
        engine.wp[idx] = (ws / denom * cfg.prop_gain).astype(np.float32)

    def count(self, engine) -> int:
        return int((engine.group == self.cfg.group).sum())

    def stats(self, engine) -> dict:
        m = engine.group == self.cfg.group
        return dict(edges=int(m.sum()), enabled=self.cfg.enabled,
                    mean_w=float(engine.w[m].mean()) if m.any() else 0.0,
                    topk_out=self.cfg.topk_out)
