# -*- coding: utf-8 -*-
"""P2 第一步：把"状态 → 动作"蒸馏成连接（不需要机器狗）。

思路（老库 14.7 的教训）：动态动作不是一串姿势，是"每一刻按身体状态修正"。
所以不抄轨迹，抄**映射**：

    示范策略：balance → 12 块肌肉的力度
    蒸馏：把"当前平衡状态激活的那些前额叶单元" → "示范动作该亮的那些运动单元"
          用监督式赫布长成连接（group 41）
    测试：撤掉示范，只喂平衡状态，看产出的力度和示范差多远（对照：没蒸馏的网络）

用法：
    python experiments/p2_distill.py --samples 300 --preset full
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import Cortex, PlasticConfig, PlasticRule, build, build_spec   # noqa: E402
from agi.build import G                                                 # noqa: E402
from agi.encoding import Observation                                    # noqa: E402

LEFT = np.array([3, 4, 5, 9, 10, 11])
RIGHT = np.array([0, 1, 2, 6, 7, 8])
FRONT = np.array([0, 1, 2, 3, 4, 5])
BACK = np.array([6, 7, 8, 9, 10, 11])


def teacher_forces(balance: np.ndarray) -> np.ndarray:
    """示范策略：右侧倾 → 左腿加力；前倾 → 后腿加力；中间 0.5 站住。"""
    roll = float(balance[0]) - 0.5
    pitch = float(balance[1]) - 0.5
    f = np.full(12, 0.5, np.float32)
    f[LEFT] += 1.2 * roll
    f[RIGHT] -= 1.2 * roll
    f[FRONT] += 0.8 * pitch
    f[BACK] -= 0.8 * pitch
    return np.clip(f, 0.0, 1.0).astype(np.float32)


def make_obs(balance: np.ndarray, spec) -> Observation:
    """只改平衡通道，其余保持恒定（隔离变量）。"""
    return Observation(vision=np.zeros((spec.px, spec.px, 3), np.float32),
                       audio=np.zeros((spec.audio_bands, spec.audio_frames), np.float32),
                       touch=np.full(spec.touch_ch, 0.3, np.float32),
                       proprio=np.full(spec.proprio_ch, 0.5, np.float32),
                       balance=np.asarray(balance, np.float32))


def motor_units_for(brain, forces: np.ndarray) -> np.ndarray:
    """示范动作 → 该点亮哪些运动单元（每块肌肉按力度档点亮前 k 个）。"""
    sub = brain.motor["sub"]
    mr1 = brain.motor["mr1"]
    out = []
    for m, f in enumerate(forces):
        k = int(round(float(f) * brain.spec.n_force))
        if k > 0:
            out.append(mr1[sub[m][:k]])
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def balance_ids(brain) -> np.ndarray:
    """平衡区 L1–L5 的全部神经元（蒸馏的"源"：状态直接来自感觉层，不过前额叶）。

    为什么不用前额叶：前额叶 top-k 2% 基本被视觉/背景占满，平衡这 10 个通道
    挤不进去，蒸馏会没有源。老库的做法也是"状态→动作一层接到运动皮层"。
    """
    return np.concatenate([brain.layers[f"balance.L{d}"].ids for d in range(1, 6)])


def state_tokens(brain, ids) -> np.ndarray:
    a = brain.engine.act_np
    return ids[a[ids] > 0.35]


def run_ticks(brain, ctx, obs, ticks: int = 4, force_ids=None) -> None:
    eng = brain.engine
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(obs))
        if force_ids is not None and len(force_ids):
            eng.set_external(force_ids, np.ones(len(force_ids), np.float32))
        eng.step()


def evaluate(brain, rng, n: int = 40) -> dict:
    """撤掉示范，只喂平衡状态，比较产出力度和示范。"""
    sp = brain.spec
    ctx = Cortex(brain, trace=False)
    preds, teachers = [], []
    for _ in range(n):
        b = np.clip(0.5 + rng.normal(0, 0.18, sp.balance_ch), 0, 1)
        obs = make_obs(b, sp)
        brain.engine.act[:] = 0.0
        brain.engine.reset_adaptation()
        run_ticks(brain, ctx, obs, ticks=8)
        preds.append(brain.forces())
        teachers.append(teacher_forces(b))
    P = np.stack(preds)
    T = np.stack(teachers)
    mae = float(np.abs(P - T).mean())
    corr = []
    for m in range(sp.n_muscle):
        if T[:, m].std() > 1e-6 and P[:, m].std() > 1e-6:
            corr.append(float(np.corrcoef(P[:, m], T[:, m])[0, 1]))
    return dict(mae=mae, corr=float(np.mean(corr)) if corr else 0.0,
                pred_std=float(P.std(axis=0).mean()),
                teacher_std=float(T.std(axis=0).mean()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--preset", default="full")
    p.add_argument("--pfc-topk", type=float, default=0.02)
    p.add_argument("--pfc-adapt", type=float, default=0.0)
    p.add_argument("--net-seed", type=int, default=0)
    args = p.parse_args(argv)

    sp = build_spec(args.preset, pfc_topk=args.pfc_topk, pfc_adapt=args.pfc_adapt,
                    seed=7 + args.net_seed)
    t0 = time.perf_counter()
    brain = build(sp, verbose=False)
    print(f"[P2] 建网 {time.perf_counter() - t0:.1f}s，"
          f"{brain.engine.n:,} 神经元 / {brain.engine.n_edges:,} 边")

    rule = PlasticRule(PlasticConfig(enabled=True, group=41, eta=0.15,
                                     eta_sim=0.0, topk_out=64, max_pairs=8192,
                                     flush_every=100000, flush_edges=12000),
                       seed=3 + args.net_seed)
    ctx = Cortex(brain, plastic=None)
    src_ids = balance_ids(brain)
    rng = np.random.default_rng(args.net_seed)

    print(f"[P2] 蒸馏：{args.samples} 个 (平衡状态 → 动作) 样本")
    grown = 0
    for i in range(args.samples):
        b = np.clip(0.5 + rng.normal(0, 0.2, sp.balance_ch), 0, 1)
        obs = make_obs(b, sp)
        tf = teacher_forces(b)
        act_ids = motor_units_for(brain, tf)
        # 让状态沿链传到前额叶，同时用示范动作顶着运动单元（teacher forcing）
        run_ticks(brain, ctx, obs, ticks=8, force_ids=act_ids)
        src = state_tokens(brain, src_ids)
        if len(src):
            grown += rule.reinforce(brain.engine, src, act_ids)
        if (i + 1) % 100 == 0:
            rule.flush(brain.engine)
            print(f"  {i + 1:4d} 样本  已写联想边 {rule.count(brain.engine):6d}")
    rule.flush(brain.engine)
    m = brain.engine.group == 41
    print(f"[P2] 蒸馏完成：group 41 边数 {int(m.sum())}，"
          f"覆盖源单元 {len(np.unique(brain.engine.src[m])) if m.any() else 0}")

    ev = evaluate(brain, rng)
    b0 = build(build_spec(args.preset, pfc_topk=args.pfc_topk,
                          pfc_adapt=args.pfc_adapt, seed=7 + args.net_seed),
               verbose=False)
    ev0 = evaluate(b0, rng)
    print("\n[P2] 测试（撤掉示范，只喂平衡状态）：")
    print(f"  蒸馏后： 力度与示范的相关系数 {ev['corr']:.3f}   MAE {ev['mae']:.3f}")
    print(f"  对照：   力度与示范的相关系数 {ev0['corr']:.3f}   MAE {ev0['mae']:.3f}")
    print(f"  （产出力度的标准差：蒸馏 {ev['pred_std']:.3f} / 对照 {ev0['pred_std']:.3f}，"
          f"示范 {ev['teacher_std']:.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
