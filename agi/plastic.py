# -*- coding: utf-8 -*-
"""联想连接（记忆）的生长规则——v0 默认关闭，边数 0。

这一层不是"外挂记忆库"，而是**前额叶神经元之间的连接**。前额叶的每个神经元
就像一个字/一个音：单独亮没有意义，靠"谁先亮、谁后亮"长出有向连接，才有关联、
才有回忆、才谈得上意识。所以：

  · 没有海马体，没有单独的 memory 模块；
  · v0 前额叶内部边 = 0（只有出生接线）；
  · 训练/测试阶段喂真实序列，调用 `PlasticRule.apply()` 让连接长出来。

**规则是"有先后的赫布"，不是"同时活跃就加强"**：

    上一拍亮 A、这一拍亮 B   →   A→B 加强        （时间顺序决定方向）
    这一拍刚亮的 B、C 之间   →   互相加强（弱）   （同一时刻的共同出现）
    一直亮 / 一直灭的不产生边                    （不携带顺序信息）
    只有"刚变化的"参与：上一拍亮、这一拍灭的 → 这一拍刚亮的

其余规则来自字神经元网络的实测（见 docs/MECHANISMS.md）：

1. **凹形生长**：Δw = η·a_i·a_j·(1 − w/w_max)^pow —— 越涨越难涨；
2. **稀疏**：每个单元只留最强的 top-k 条出边；
3. **整体降权**：每 N 拍 ×decay，让"从不用"的连接消失；
4. **两层权重**：连接强度 W ≠ 传播权重 Ŵ（扇出归一 + 枢纽抑制 + 锐化）；
5. **特异性闸门（PMI）**：压掉"什么都能连"的万金油连接。

实现说明：为了能在"每拍上万对"的规模上跑，这里全部向量化——
存在边用**有序 key 表 + 二分查找**，共现计数用**与边对齐的数组**，不逐对做字典查询。
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
    in_floor: float = 8.0           # 入边和的"免罚下限"：低于它不压，高于它按比例压枢纽
    out_floor: float = 1e-6         # 出边和下限（防除零）
    # 特异性闸门（PMI）：压掉"什么都能连"的万金油连接
    pmi_gate: bool = True
    pmi_steep: float = 1.2          # 字神经元网络实测的陡度
    pmi_offset: float = 0.0         # PMI 零点
    pmi_floor: float = 0.12         # 闸门下限（再差也留 12%）
    pmi_decay_every: int = 500      # 多久把共现计数衰减一次（滑动窗口）
    pmi_decay: float = 0.97


class PlasticRule:
    """前额叶内的顺序赫布生长。v0 里 `enabled=False`，`apply()` 是空操作。"""

    def __init__(self, cfg: PlasticConfig | None = None, seed: int = 3):
        self.cfg = cfg or PlasticConfig()
        self.rng = np.random.default_rng(seed)
        # 待写入缓冲区（每拍攒着，flush 时批量写进引擎）
        self._ns: list[np.ndarray] = []
        self._nd: list[np.ndarray] = []
        self._nw: list[np.ndarray] = []
        self._nc: list[np.ndarray] = []
        self._pending_n = 0
        # 已写入边的索引（按 key 排序）与每条边的共现计数
        self._keys = np.zeros(0, np.int64)
        self._kid = np.zeros(0, np.int64)
        self._cnt_edge = np.zeros(0, np.float32)
        self._cnt_unit = np.zeros(0, np.float32)
        self._prev = np.zeros(0, np.int64)
        self._grown = 0
        self._strengthened = 0
        self.t = 0

    # ------------------------------------------------------------ 主入口
    def apply(self, engine, cur_active, prev_active=None) -> dict:
        if not self.cfg.enabled:
            return dict(edges=0, grown=0, strengthened=0, pruned=0)
        cur = np.unique(np.asarray(cur_active, np.int64))
        prev = self._prev if prev_active is None else np.unique(
            np.asarray(prev_active, np.int64))
        if self._cnt_unit.size != engine.n:
            self._cnt_unit = np.zeros(engine.n, np.float32)
        if len(cur):
            np.add.at(self._cnt_unit, cur, 1.0)
        grown = strengthened = 0
        if len(prev) and len(cur):
            grown, strengthened = self._hebb(engine, prev, cur)
        self.t += 1
        pruned = 0
        if self._pending_n and (self._pending_n >= self.cfg.flush_edges
                                or self.t % max(1, self.cfg.flush_every) == 0):
            pruned = self.flush(engine)
        if self.cfg.pmi_decay_every and self.t % self.cfg.pmi_decay_every == 0:
            self._cnt_unit *= self.cfg.pmi_decay
            self._cnt_edge *= self.cfg.pmi_decay
        self._prev = cur
        return dict(edges=self.count(engine), grown=grown,
                    strengthened=strengthened, pruned=pruned,
                    pending=self._pending_n)

    # ------------------------------------------------------------ 赫布
    def _hebb(self, engine, prev: np.ndarray, cur: np.ndarray) -> tuple[int, int]:
        cfg = self.cfg
        self._grown = self._strengthened = 0
        # 同一画面持续时不学：那不是"先 A 后 B"，只是"还停在 A"。
        if len(cur):
            overlap = len(np.intersect1d(prev, cur, assume_unique=True)) / len(cur)
            if overlap > 1.0 - cfg.min_change:
                return 0, 0
        n = engine.n
        mask_prev = np.zeros(n, bool)
        mask_prev[prev] = True
        mask_cur = np.zeros(n, bool)
        mask_cur[cur] = True
        newly = cur[~mask_prev[cur]]        # 这一拍"刚亮"的
        dropped = prev[~mask_cur[prev]]     # 上一拍亮、这一拍灭的（"先亮的"）
        # 只有"刚变化的"参与（链式库 plastic2 的规则）。不这样做，边会大量指回自己。
        pre = dropped if len(dropped) else prev
        post = newly if len(newly) else cur
        n_all = len(pre) * len(post)
        if n_all <= cfg.full_cross:
            ii = np.repeat(pre, len(post))
            jj = np.tile(post, len(pre))
            keep = ii != jj
            self._apply_pairs(engine, ii[keep], jj[keep], cfg.eta)
        else:
            n_pair = min(cfg.max_pairs, n_all)
            ii = pre[self.rng.integers(0, len(pre), n_pair)]
            jj = post[self.rng.integers(0, len(post), n_pair)]
            keep = ii != jj
            self._apply_pairs(engine, ii[keep], jj[keep], cfg.eta)
        # 同拍刚亮：互相加强（双向、弱）
        if len(newly) > 1:
            k = min(cfg.max_pairs // 2, len(newly) * max(1, len(newly) - 1) // 2)
            if k > 0:
                a = newly[self.rng.integers(0, len(newly), k)]
                b = newly[self.rng.integers(0, len(newly), k)]
                keep = a != b
                self._apply_pairs(engine, a[keep], b[keep], cfg.eta_sim)
        return self._grown, self._strengthened

    def _apply_pairs(self, engine, src: np.ndarray, dst: np.ndarray, eta: float) -> None:
        """向量化：已写入的就地加强，新的先攒着。"""
        if not len(src):
            return
        n = engine.n
        keys, cnt = np.unique(src.astype(np.int64) * n + dst.astype(np.int64),
                              return_counts=True)
        # 已存在的边：二分查找
        if len(self._keys):
            pos = np.searchsorted(self._keys, keys)
            pos = np.minimum(pos, len(self._keys) - 1)
            found = self._keys[pos] == keys
        else:
            pos = np.zeros(len(keys), np.int64)
            found = np.zeros(len(keys), bool)
        if found.any():
            ei = self._kid[pos[found]]
            w = engine.w[ei]
            inc = eta * np.maximum(0.0, 1.0 - w / self.cfg.w_max) ** self.cfg.pow
            engine.w[ei] += inc.astype(np.float32)
            self._cnt_edge[ei] += cnt[found]
            self._strengthened += int(found.sum())
        if (~found).any():
            nk = keys[~found]
            self._ns.append((nk // n).astype(np.int32))
            self._nd.append((nk % n).astype(np.int32))
            self._nw.append(np.full(len(nk), eta, np.float32))
            self._nc.append(cnt[~found].astype(np.float32))
            self._pending_n += len(nk)
            self._grown += int((~found).sum())

    # ------------------------------------------------------------ 写入
    def flush(self, engine) -> int:
        """把攒着的新边批量写进引擎（会重建后端张量，所以别每拍都做）。"""
        if not self._pending_n:
            return 0
        n = engine.n
        s = np.concatenate(self._ns)
        d = np.concatenate(self._nd)
        w = np.concatenate(self._nw)
        c = np.concatenate(self._nc)
        keys = s.astype(np.int64) * n + d.astype(np.int64)
        uniq, inv = np.unique(keys, return_inverse=True)
        ww = np.zeros(len(uniq), np.float32)
        np.maximum.at(ww, inv, w)                   # 同一对重复出现时取最大权重
        cc = np.zeros(len(uniq), np.float32)
        np.add.at(cc, inv, c)
        engine.add_edges((uniq // n).astype(np.int32), (uniq % n).astype(np.int32),
                         ww, group=self.cfg.group)
        self._cnt_edge = np.concatenate([self._cnt_edge, cc])
        self._ns, self._nd, self._nw, self._nc = [], [], [], []
        self._pending_n = 0
        self._rebuild_index(engine)
        self._prune_out(engine)
        pruned = 0
        if self.cfg.decay_every and self.t % self.cfg.decay_every == 0:
            self._decay(engine)
            pruned = self._prune(engine)
        self.sync(engine)
        engine.rebuild_backend()
        return pruned

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
        drop = rank >= cfg.topk_out
        if drop.any():
            keep_local = ~drop[np.argsort(order)]
            self._drop(engine, idx, keep_local)

    def _prune(self, engine) -> int:
        idx = np.flatnonzero(engine.group == self.cfg.group)
        if not len(idx):
            return 0
        keep_local = engine.w[idx] >= self.cfg.min_w
        if (~keep_local).any():
            self._drop(engine, idx, keep_local)
        return int((~keep_local).sum())

    def _drop(self, engine, idx: np.ndarray, keep_local: np.ndarray) -> None:
        mask = np.zeros(len(engine.src), bool)
        mask[idx[~keep_local]] = True
        engine.drop_edges(mask)
        self._cnt_edge = self._cnt_edge[keep_local]
        self._rebuild_index(engine)

    def _decay(self, engine) -> None:
        m = engine.group == self.cfg.group
        if m.any():
            engine.w[m] *= self.cfg.decay

    def _rebuild_index(self, engine) -> None:
        idx = np.flatnonzero(engine.group == self.cfg.group)
        keys = engine.src[idx].astype(np.int64) * engine.n + engine.dst[idx]
        order = np.argsort(keys)
        self._keys = keys[order]
        self._kid = idx[order]

    # ------------------------------------------------------------ 传播权重
    def gates(self, engine, idx=None) -> np.ndarray:
        """每条边的特异性闸门（0.12..1）。万金油连接的 PMI 低，被压到下限。"""
        cfg = self.cfg
        if idx is None:
            idx = np.flatnonzero(engine.group == cfg.group)
        T = max(1.0, float(self.t))
        cab = self._cnt_edge if len(self._cnt_edge) == len(idx) else np.zeros(len(idx), np.float32)
        pa = self._cnt_unit[engine.src[idx]] / T
        pb = self._cnt_unit[engine.dst[idx]] / T
        eps = 1e-6
        pmi = np.log((cab / T + eps) / ((pa + eps) * (pb + eps)))
        return (cfg.pmi_floor + (1.0 - cfg.pmi_floor) / (
            1.0 + np.exp(-cfg.pmi_steep * (pmi - cfg.pmi_offset)))).astype(np.float32)

    def sync(self, engine) -> None:
        """连接强度 W → 传播权重 Ŵ。

        和字神经元网络有一处**有意的偏差**：原库同时除以出边和与入边和，
        因为那里的 W 是唯一驱动。我们这里前额叶还有一条 1:1 的感觉线，
        两份归一叠起来会把联想压到 1%，等于白学。所以只保留扇出归一 + 枢纽抑制。
        """
        cfg = self.cfg
        idx = np.flatnonzero(engine.group == cfg.group)
        if not len(idx):
            return
        w = np.maximum(engine.w[idx], 0.0)
        out = np.bincount(engine.src[idx], weights=w, minlength=engine.n)
        inn = np.bincount(engine.dst[idx], weights=w, minlength=engine.n)
        ws = w ** cfg.sharpen
        denom = np.maximum(out[engine.src[idx]], cfg.out_floor)
        hub = np.maximum(inn[engine.dst[idx]], cfg.in_floor) / cfg.in_floor
        wp = ws / denom / hub * cfg.prop_gain
        if cfg.pmi_gate and self._cnt_unit.size == engine.n and self.t > 5:
            wp = wp * self.gates(engine, idx)
        engine.wp[idx] = wp.astype(np.float32)

    def count(self, engine) -> int:
        return int((engine.group == self.cfg.group).sum())

    def stats(self, engine) -> dict:
        m = engine.group == self.cfg.group
        return dict(edges=int(m.sum()), enabled=self.cfg.enabled,
                    mean_w=float(engine.w[m].mean()) if m.any() else 0.0,
                    topk_out=self.cfg.topk_out)
