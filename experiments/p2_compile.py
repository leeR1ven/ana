# -*- coding: utf-8 -*-
"""P2·甲：把示范策略"抄"成连接（compile，不是学）。

做法：
  1. 状态网格上采样（roll × pitch），跑网络，记下
       A = 平衡区 L1–L5 的活跃向量、d0 = 力度细胞当前的驱动、k = 示范的力度档；
  2. 每个力度档细胞（肌肉 m、第 l 档）当一个小分类器：
       标签 y = (k_m ≥ l)
       权重 w = max(0, log P(单元亮 | y=1) / P(单元亮 | y=0))    （只取正证据）
       打分 = d0 + w·A，门槛取"训练集上分对最多"的那个点；
  3. 把 w 写成 状态单元 → 力度细胞 的边（group 42），门槛写进细胞；
  4. 在**没参与编译**的状态上验收：能不能复现示范。

用法：
    python experiments/p2_compile.py --preset full --grid 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec                                       # noqa: E402
from experiments.p2_distill import (LEFT, RIGHT, FRONT, BACK, make_obs,   # noqa: E402
                                    teacher_forces)

G_STATE2FORCE = 42


def balance_layer_ids(brain) -> np.ndarray:
    return np.concatenate([brain.layers[f"balance.L{d}"].ids for d in range(1, 6)])


def run_state(brain, obs, ticks: int = 12):
    """跑一个状态，返回 (状态活跃向量, 力度细胞驱动)。"""
    eng = brain.engine
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(obs))
        eng.step()
    return eng.act_np.copy()


def sample_states(n_grid: int, rng, n_random: int = 40) -> np.ndarray:
    """网格 + 随机：平衡通道 0=roll、1=pitch 变化，其余 0.5。"""
    g = np.linspace(0.05, 0.95, n_grid)
    out = []
    for r in g:
        for p in g:
            out.append((r, p))
    for _ in range(n_random):
        out.append((rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)))
    return np.asarray(out, np.float32)


def make_balance(rp, spec) -> np.ndarray:
    b = np.full(spec.balance_ch, 0.5, np.float32)
    b[0], b[1] = rp
    return b


def collect(brain, states, spec, ticks=12):
    ids = balance_layer_ids(brain)
    force_ids = np.concatenate([brain.motor["force"][m] for m in range(spec.n_muscle)])
    A, D, K = [], [], []
    for rp in states:
        bal = make_balance(rp, spec)
        brain.engine.act[:] = 0.0        # 每个状态从零起，和测试时一致
        act = run_state(brain, make_obs(bal, spec), ticks)
        A.append(act[ids])
        D.append(brain.engine.drive().numpy()[force_ids])
        K.append(np.rint(teacher_forces(bal) * spec.n_force).astype(int))
    return ids, force_ids, np.stack(A), np.stack(D), np.stack(K)


def compile_cells(A, D, K, n_force, min_pos=3, max_edges=40):
    """每个力度档细胞一个正证据分类器；返回 (权重矩阵, 门槛, 每格准确率)。"""
    n = len(A)
    W = np.zeros((n_force * 12, A.shape[1]), np.float32)
    thr = np.full(n_force * 12, np.inf, np.float32)
    acc = np.zeros(n_force * 12, np.float32)
    for m in range(12):
        for l in range(n_force):
            j = m * n_force + l
            y = (K[:, m] >= (l + 0.5)).astype(np.int8)
            if y.sum() < min_pos or (1 - y).sum() < min_pos:
                continue
            p1 = A[y == 1].mean(0)
            p0 = A[y == 0].mean(0)
            w = np.maximum(0.0, np.log((p1 + 1e-3) / (p0 + 1e-3))).astype(np.float32)
            if w.max() <= 0:
                continue
            w *= 1.0 / w.max()                      # 归一，门槛在同一量纲里选
            score = D[:, j] + A @ w
            # 在候选门槛里挑分对最多的
            cand = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 49)))
            best, bt = -1, np.inf
            for t in cand:
                a = float(((score > t) == (y == 1)).mean())
                if a > best:
                    best, bt = a, float(t)
            W[j] = w
            thr[j] = bt
            acc[j] = best
    return W, thr, acc


def install(brain, ids, force_ids, W, thr, topk: int = 40):
    eng = brain.engine
    src, dst, w = [], [], []
    for j, row in enumerate(W):
        if not np.isfinite(thr[j]):
            continue
        nz = np.flatnonzero(row > 0)
        if len(nz) > topk:
            nz = nz[np.argsort(-row[nz])[:topk]]
        src.append(ids[nz])
        dst.append(np.full(len(nz), force_ids[j], np.int64))
        w.append(row[nz])
    if src:
        eng.add_edges(np.concatenate(src).astype(np.int32),
                      np.concatenate(dst).astype(np.int32),
                      np.concatenate(w).astype(np.float32), group=G_STATE2FORCE)
    eng.set_thr(force_ids, thr=thr.astype(np.float32),
                soft=np.full(len(force_ids), 0.15, np.float32))
    eng.rebuild_backend()


def evaluate(brain, states, spec, ticks=12):
    preds, teachers = [], []
    for rp in states:
        bal = make_balance(rp, spec)
        brain.engine.act[:] = 0.0
        run_state(brain, make_obs(bal, spec), ticks)
        preds.append(brain.forces())
        teachers.append(teacher_forces(bal))
    P, T = np.stack(preds), np.stack(teachers)
    mae = float(np.abs(P - T).mean())
    corr = []
    for m in range(spec.n_muscle):
        if T[:, m].std() > 1e-6 and P[:, m].std() > 1e-6:
            corr.append(float(np.corrcoef(P[:, m], T[:, m])[0, 1]))
    exact = float((np.abs(P - T) < 0.051).mean())
    return dict(mae=mae, corr=float(np.mean(corr)) if corr else 0.0, exact=exact,
                pred_std=float(P.std(0).mean()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="full")
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--net-seed", type=int, default=0)
    p.add_argument("--ticks", type=int, default=12)
    args = p.parse_args(argv)

    sp = build_spec(args.preset, pfc_topk=0.02, seed=7 + args.net_seed)
    brain = build(sp, verbose=False)
    rng = np.random.default_rng(args.net_seed)
    train = sample_states(args.grid, rng, n_random=40)
    test = sample_states(args.grid + 1, np.random.default_rng(999), n_random=60)
    # 留出：去掉与训练重合的网格点
    test = test[~np.isin(test[:, 0], train[:, 0])]

    print(f"[甲] 编译：{len(train)} 个训练状态、{len(test)} 个留出状态")
    ids, force_ids, A, D, K = collect(brain, train, sp, args.ticks)
    W, thr, acc = compile_cells(A, D, K, sp.n_force)
    print(f"  可编译的力度细胞 {int(np.isfinite(thr).sum())} / {len(thr)}，"
          f"训练集准确率均值 {float(acc[np.isfinite(thr)].mean()):.3f}")

    before = evaluate(brain, test, sp, args.ticks)
    install(brain, ids, force_ids, W, thr)
    after = evaluate(brain, test, sp, args.ticks)
    print("\n[甲] 留出状态上的验收：")
    print(f"  编译前：相关 {before['corr']:.3f}  MAE {before['mae']:.3f}  "
          f"逐档命中 {before['exact']:.3f}")
    print(f"  编译后：相关 {after['corr']:.3f}  MAE {after['mae']:.3f}  "
          f"逐档命中 {after['exact']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
