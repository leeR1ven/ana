# -*- coding: utf-8 -*-
"""扩维去相关层（模仿齿状回）。

问题：两段经历的感觉码本来就有 20%~50% 重叠，直接拿去做联想记忆，写进去的边
一强就会把两段拉成一个吸引子（实测"独有单元只剩 8 个"）。

齿状回的做法：**扩维 + 稀疏 + 强抑制竞争**。相似输入在这层被拆成几乎不重叠的集合，
之后的联想记忆才有"分得开"的基础。

实现：从感觉 L5 随机稀疏连接到 4 倍宽的单元，每拍只保留响应最强的 2%（k-WTA）。
"""
from __future__ import annotations

import numpy as np


def build_dentate(engine, src_ids, factor: int = 4, k_in: int = 4,
                  active_frac: float = 0.02, group: int = 66, seed: int = 13,
                  region: str = "time"):
    """建一层扩维去相关单元，返回它们的编号。"""
    src = np.asarray(src_ids, np.int64)
    n = int(len(src) * factor)
    pos = np.stack([np.full(n, 0.97, np.float32),
                    np.linspace(0, 1, n, dtype=np.float32),
                    np.full(n, 0.5, np.float32)], 1)
    if getattr(engine, "_final", False):
        ids = engine.grow_units(n, kind="pfc", region=region, depth=0,
                                thr=0.0, soft=1.0, pos=pos, counted=False)
    else:
        ids = engine.add_block(n, kind="pfc", region=region, depth=0,
                               thr=0.0, soft=1.0, pos=pos, counted=False)
    rng = np.random.default_rng(seed)
    # 每个扩维单元随机读 k_in 个感觉单元（权重 ~1/k_in）
    pick = src[rng.integers(0, len(src), size=(n, k_in))]
    _w = np.full(n * k_in, 1.0 / k_in, np.float32)
    if getattr(engine, "_final", False):
        engine.add_edges(np.repeat(ids, k_in).astype(np.int32),
                         pick.reshape(-1).astype(np.int32), _w, group=group)
    else:
        engine.connect(np.repeat(ids, k_in), pick.reshape(-1), _w, group=group)
    engine.set_thr(ids, thr_frac=0.25, soft_frac=0.2)
    k = max(1, int(round(active_frac * n)))
    engine.register_topk(ids, k)          # 名额竞争：每拍只留最强 2%（确定性靠 _prio）
    if getattr(engine, "_final", False):
        engine.rebuild_backend()          # 构建期不用（finalize 时才统一建张量）
    return ids


def dentate_code(engine, ids) -> np.ndarray:
    """返回活跃单元的**全局编号**（不是局部下标）。"""
    ids = np.asarray(ids)
    a = np.asarray(engine.act)[ids]
    return ids[np.flatnonzero(a > 0.5)]
