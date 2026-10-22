# -*- coding: utf-8 -*-
"""架构自检：只查"架构上一定不成立"的事，不查效果。

`check()` 是静态不变量（不跑网络）；`check_behavior()` 跑十几拍看通路是否通。
整合/改结构之后必须先跑这两个。任何一条 issue 都意味着某个不变量被破坏了。
"""
from __future__ import annotations

import numpy as np

from .build import G


def check(brain, expect_no_plastic: bool = True, verbose: bool = False) -> list[str]:
    eng = brain.engine
    sp = brain.spec
    n = eng.n
    issues: list[str] = []

    def add(msg: str):
        issues.append(msg)

    # ---------------------------------------------------------- 基本
    if len(eng.src) and (eng.src.min() < 0 or eng.dst.min() < 0
                         or eng.src.max() >= n or eng.dst.max() >= n):
        add("边端点越界")
    if len(eng.src) and (eng.src == eng.dst).any():
        add("存在自环边")
    if len(eng.src):
        key = eng.src.astype(np.int64) * n + eng.dst
        if len(np.unique(key)) != len(key):
            add(f"存在重复边 {len(key) - len(np.unique(key))} 条")
        if np.any(eng.w < 0):
            add("存在负权重（抑制应该用抑制细胞，不在权重上写负号）")

    has_in = np.bincount(eng.dst, minlength=n) > 0
    n_in = eng.input_size

    # ---------------------------------------------------------- 输入
    if len(eng.src) and (eng.dst < n_in).any():
        add("感觉细胞有入边（感觉层必须只由外界驱动）")

    # ---------------------------------------------------------- 每层
    for name, L in brain.layers.items():
        if L.k <= 0 or L.k > L.size:
            add(f"{name}: top-k {L.k} 不在 1..{L.size}")
        if int(L.ids[-1]) - int(L.ids[0]) + 1 != L.size:
            add(f"{name}: 编号不是连续块（top-k 需要连续块）")
        bad = brain.filtered.get(name)
        kept = (~bad) if bad is not None and len(bad) == L.size else np.ones(L.size, bool)
        if not has_in[L.ids[kept]].all():
            add(f"{name}: {int((~has_in[L.ids[kept]]).sum())} 个神经元没有入边")
        if bad is not None and bad.any():
            # 专用线是"从该细胞出去"的 group 30 边；被筛掉的不能还有
            if (np.isin(eng.src, L.ids[bad]) & (eng.group == G.PFC)).any():
                add(f"{name}: 被筛掉的神经元还接着专用线")
        key = (L.region, L.depth)
        if key not in brain.ids["relay"]:
            add(f"{name}: 缺专用线中继")
            continue
        if key not in brain.pfc_groups:
            add(f"{name}: 缺前额叶组")
            continue
        r1, r2 = brain.ids["relay"][key]
        gf = eng.group == G.PFC
        f1 = np.zeros(n, bool)
        f1[r1] = True
        f2 = np.zeros(n, bool)
        f2[r2] = True
        want = int(kept.sum())
        if int((f1[eng.dst] & gf).sum()) != want:
            add(f"{name}: 到第一级中继的专用线 {int((f1[eng.dst] & gf).sum())} != {want}")
        if int((f2[eng.dst] & gf).sum()) != want:
            add(f"{name}: 第二级中继入线 {int((f2[eng.dst] & gf).sum())} != {want}")
        # 回程：只在 L2..L5，且 ≤ 前向一半
        fl = np.zeros(n, bool)
        fl[L.ids] = True
        fwd = fl[eng.src]
        ret = fwd & (eng.group == G.RET_L2)
        if L.depth >= 2 and int(ret.sum()) > 0.5 * int(fwd.sum()):
            add(f"{name}: 回程 {int(ret.sum())} > 前向一半 {int(fwd.sum()) // 2}")
        if L.depth == 1 and int(ret.sum()):
            add(f"{name}: 回程落回了 L1")

    # ---------------------------------------------------------- 温度计
    if "vision" in brain.ids["input"]:
        from .encoding import thermometer
        cells = brain.ids["input"]["vision"]        # (C, 2, N)
        C, _, N = cells.shape
        pos, neg = thermometer(np.full(C, 0.37, np.float32), N)
        n_on = np.concatenate([pos, neg], axis=1).sum(axis=1)
        if not np.all(n_on == N):
            add(f"温度计不是每对恰好亮一个（每通道 {np.unique(n_on)[:5]}）")

    # ---------------------------------------------------------- 前额叶 / 记忆
    n_pfc_internal = int((eng.group == G.PL_PFC).sum())
    if expect_no_plastic and n_pfc_internal:
        add(f"v0 前额叶内部应有 0 条边，实际 {n_pfc_internal}")
    n_plastic = int((eng.group == G.PLASTIC).sum())
    if expect_no_plastic and n_plastic:
        add(f"v0 可塑边应为 0，实际 {n_plastic}")
    for key, g in brain.pfc_groups.items():
        name = f"{key[0]}.L{key[1]}"
        bad = brain.filtered.get(name)
        kept_g = (~bad) if bad is not None and len(bad) == len(g) else np.ones(len(g), bool)
        if not has_in[g[kept_g]].all():
            add(f"前额叶组 {key} 有 {int((~has_in[g[kept_g]]).sum())} 个神经元没有入线")
        if bad is not None and bad.any() and has_in[g[bad]].any():
            add(f"前额叶组 {key}: 被筛掉的单元还有入线")
    if not np.isfinite(eng.thr[brain.pfc_ids]).all():
        add("前额叶门槛出现 NaN/Inf")

    # ---------------------------------------------------------- 运动区
    m = brain.motor
    if len(m["mpfc"]) < sp.n_muscle:
        add("运动相关组比肌肉数还少")
    for i in range(sp.n_muscle):
        if len(m["force"][i]) != sp.n_force:
            add(f"肌肉 {i} 的力度神经元数不是 {sp.n_force}")
        if not has_in[m["force"][i]].all():
            add(f"肌肉 {i} 有力度神经元没有输入")
    thr_all = np.stack([eng.thr[m["force"][i]] for i in range(sp.n_muscle)])
    if not np.all(np.diff(thr_all, axis=1) >= 0):
        add("力度档门槛不是单调递增（分档会乱）")
    g_i = eng.group == G.MOTOR_BLOCK
    if g_i.any() and not np.all(eng.sgn[g_i] < 0):
        add("I → 力度 的边不是抑制")
    g_d = eng.group == G.MOTOR_DEINH
    if g_d.any() and not np.all(eng.sgn[g_d] < 0):
        add("D → I 的边不是抑制")
    if g_i.sum() == 0:
        add("没有 I → 力度 的门控边")
    # 区与区直连：必须真的跨区，而且不能落在可塑组里（天生不可改）
    g_rl = eng.group == G.REGION_LINK
    if g_rl.any():
        if (eng.region[eng.src[g_rl]] == eng.region[eng.dst[g_rl]]).any():
            add("区与区直连里出现了同一个区内部的边")
        if (eng.group[g_rl] == G.PL_PFC).any():
            add("区与区直连被放进了可塑组（必须不可改）")

    if verbose:
        print(f"[check] {len(issues)} issues / {n:,} neurons / {eng.n_edges:,} edges")
        for i in issues:
            print("  -", i)
    return issues


def check_behavior(brain, ticks: int = 16) -> list[str]:
    """跑一小段闭环：前额叶要有活动、出力要有限、每拍要在预算内。"""
    from .loop import Cortex
    from .world import sample_observation, VirtualDog
    issues: list[str] = []
    sp = brain.spec
    ctx = Cortex(brain, trace=False)
    dog = VirtualDog(sp, seed=2)
    obs = dog.observe()
    pfc_max = 0
    for i in range(ticks):
        tk = ctx.tick(obs, record=False)
        pfc_max = max(pfc_max, tk.pfc_active)
        if not np.all(np.isfinite(tk.forces)):
            issues.append(f"第 {i} 拍出力出现 NaN")
            break
        if not np.all((tk.forces >= 0) & (tk.forces <= 1)):
            issues.append(f"第 {i} 拍出力越界 {tk.forces}")
            break
        dog.step(tk.forces)
        obs = dog.observe()
    if pfc_max == 0:
        issues.append("跑完一整段前额叶都没亮过（连线/门槛/标定有问题）")
    # 单拍耗时（含编码 + 步进 + 读运动区）
    b = ctx.bench(ticks=10)
    if b["mean_ms"] > b["budget_ms"]:
        issues.append(f"单拍 {b['mean_ms']:.1f} ms 超过预算 {b['budget_ms']:.1f} ms")
    return issues


def assert_ok(brain, behavior: bool = True, **kw) -> None:
    issues = check(brain, **kw)
    if behavior:
        issues += check_behavior(brain)
    if issues:
        raise AssertionError("架构自检失败：\n  - " + "\n  - ".join(issues))
