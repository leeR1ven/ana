# -*- coding: utf-8 -*-
"""P1 基线：打开"顺序赫布"，看前额叶内部连接长不长得出来、有没有用。

用法：
    python experiments/p1_memory.py --ticks 600 --preset xl
    python experiments/p1_memory.py --ticks 200 --preset tiny   # 冒烟

三件事：
  1. 生长曲线：边数、平均权重、每拍前额叶活跃数；
  2. 上下文依赖：同一帧在不同前序下，前额叶码是否不同（有记忆才会不同）；
  3. 自持：同一帧重复喂，前额叶活动是否靠内部连接维持（而不是纯喂进去）。

对照组：同一个网络，plastic 关闭，喂完全相同的刺激。
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
from agi.world import VirtualDog, _blob                                 # noqa: E402


# ---------------------------------------------------------------- 刺激
def stim(kind: str, spec) -> Observation:
    """三种固定画面（左块 / 中块 / 右块）+ 各自的音调。按 spec 的通道数生成。"""
    px = spec.px
    cx = {"left": 0.25, "mid": 0.5, "right": 0.75}[kind]
    img = np.zeros((px, px, 3), np.float32) + 0.05
    img = np.maximum(img, _blob(px, cx, 0.5, 0.18, (0.9, 0.2, 0.15)))
    au = np.zeros((spec.audio_bands, spec.audio_frames), np.float32)
    band = int({"left": 0.2, "mid": 0.5, "right": 0.8}[kind] * spec.audio_bands)
    au[max(0, band - 1):band + 2] = 0.9
    return Observation(vision=img, audio=au,
                       touch=np.zeros(spec.touch_ch, np.float32),
                       proprio=np.zeros(spec.proprio_ch, np.float32),
                       balance=np.full(spec.balance_ch, 0.5, np.float32))


def run_sequence(ctx: Cortex, kinds: list[str], spec, hold: int = 12) -> list[np.ndarray]:
    codes = []
    for k in kinds:
        obs = stim(k, spec)
        for _ in range(hold):
            ctx.tick(obs, record=False)
        codes.append(ctx.brain.pfc_code().copy())
    return codes


def scripted_forces(t: int, n: int) -> np.ndarray:
    """原地小步 + 缓慢左右转：画面会周期性扫过视野，产生有先后结构的经历。"""
    f = np.full(n, 0.35, np.float32)
    turn = 0.5 + 0.5 * np.sin(t * 0.05)
    f[: n // 2] *= 0.6 + 0.4 * turn
    f[n // 2:] *= 1.0 - 0.4 * turn
    return f


# ---------------------------------------------------------------- 训练
def train(ticks: int, preset: str, seed: int = 0, verbose: bool = True,
          pfc_topk: float = 0.0, protocol: str = "stim", pfc_adapt: float = 0.0,
          net_seed: int = 0):
    sp = build_spec(preset, pfc_topk=pfc_topk, pfc_adapt=pfc_adapt,
                    seed=7 + net_seed)
    b = build(sp, verbose=False)
    rule = PlasticRule(PlasticConfig(enabled=True, eta=0.10, eta_sim=0.02,
                                     topk_out=8, max_pairs=2048,
                                     decay_every=1000, decay=0.995,
                                     flush_every=100, flush_edges=20000),
                       seed=3 + net_seed)
    ctx = Cortex(b, plastic=rule)
    dog = VirtualDog(sp, seed=seed)
    cycle = ["left", "mid", "right", "mid"]
    hold = 8
    log = []
    t0 = time.perf_counter()
    for t in range(ticks):
        if protocol == "stim":
            obs = stim(cycle[(t // hold) % len(cycle)], sp)
        else:
            obs = dog.observe()
        tk = ctx.tick(obs)
        if protocol != "stim":
            dog.step(scripted_forces(t, sp.n_muscle))
        if t % max(1, ticks // 10) == 0 or t == ticks - 1:
            m = b.engine.group == G.PL_PFC
            w = b.engine.w[m]
            log.append(dict(t=t, edges=int(m.sum()),
                            mean_w=float(w.mean()) if len(w) else 0.0,
                            pfc=int(tk.pfc_active)))
            if verbose:
                print(f"  t={t:5d}  联想边 {log[-1]['edges']:6d}  "
                      f"平均权重 {log[-1]['mean_w']:.4f}  前额叶 {log[-1]['pfc']:6d}")
    dt = time.perf_counter() - t0
    if verbose:
        print(f"  训练 {ticks} 拍 / {dt:.1f}s = {ticks / dt:.1f} 拍/秒")
    return b, ctx, log


# ---------------------------------------------------------------- 测验
def context_test(brain, kind: str = "trained", hold: int = 10) -> float:
    """同一帧 B，前面接 A 或接 C：两个前额叶码的 Jaccard 距离。"""
    ctx = Cortex(brain, trace=False)
    a = run_sequence(ctx, ["left", "mid"], brain.spec, hold)[-1]
    c = run_sequence(ctx, ["right", "mid"], brain.spec, hold)[-1]
    sa, sc = set(a.tolist()), set(c.tolist())
    if not sa or not sc:
        return 0.0
    inter = len(sa & sc)
    union = len(sa | sc)
    return 1.0 - inter / union


def recall_test(brain, hold: int = 10) -> dict:
    """先看 A 再看 B：B 的码里是否多出 A 的单元（联想召回）。"""
    ctx = Cortex(brain, trace=False)

    def reset():
        brain.engine.act[:] = 0.0

    # 冷启动：直接看 B
    reset()
    b_cold = run_sequence(ctx, ["mid"], brain.spec, hold)[-1]
    # 带前序：先看 A 再看 B
    reset()
    a_code = run_sequence(ctx, ["left"], brain.spec, hold)[-1]
    b_cued = run_sequence(ctx, ["mid"], brain.spec, hold)[-1]
    A = set(a_code.tolist())
    cold = len(A & set(b_cold.tolist())) / max(1, len(A))
    cued = len(A & set(b_cued.tolist())) / max(1, len(A))
    return dict(cold=cold, cued=cued, lift=cued - cold)


def persistence(brain, hold: int = 24) -> float:
    """同一帧反复喂：前额叶活跃数（有内部连接时会因历史而抬升/自持）。"""
    ctx = Cortex(brain, trace=False)
    obs = stim("mid", brain.spec)
    counts = []
    for _ in range(hold):
        counts.append(ctx.tick(obs, record=False).pfc_active)
    head = float(np.mean(counts[2:6]))
    tail = float(np.mean(counts[-6:]))
    return tail / max(head, 1.0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=600)
    p.add_argument("--preset", default="xl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--net-seed", type=int, default=0, help="网络接线/采样的种子")
    p.add_argument("--pfc-topk", type=float, default=0.02,
                   help="前额叶动态抑制：每拍只留最强的这个比例（0=关）")
    p.add_argument("--protocol", default="stim", choices=("stim", "dog"),
                   help="训练流：stim=固定刺激序列（可测召回）；dog=身体乱转")
    p.add_argument("--pfc-adapt", type=float, default=0.01,
                   help="前额叶自适应速率（0=关）")
    args = p.parse_args(argv)

    print(f"[P1] 训练：{args.preset}，{args.ticks} 拍，顺序赫布开启，"
          f"前额叶动态抑制 {args.pfc_topk:.0%}，自适应 {args.pfc_adapt}")
    b, ctx, log = train(args.ticks, args.preset, args.seed,
                        pfc_topk=args.pfc_topk, protocol=args.protocol,
                        pfc_adapt=args.pfc_adapt, net_seed=args.net_seed)
    m = b.engine.group == G.PL_PFC
    print(f"\n[P1] 结果：联想边 {int(m.sum())} 条，"
          f"覆盖前额叶 {len(np.unique(b.engine.src[m])) if m.any() else 0} 个源单元")

    print("\n[P1] 对照：同一网络，plastic 关闭")
    b0 = build(build_spec(args.preset, pfc_topk=args.pfc_topk,
                          pfc_adapt=args.pfc_adapt,
                          seed=7 + args.net_seed), verbose=False)
    d_on = context_test(b)
    d_off = context_test(b0)
    print(f"  上下文依赖（同帧不同前序的码距离）：开学习 {d_on:.3f}   关学习 {d_off:.3f}")
    print(f"  重复同帧的自持比（尾/头）：开学习 {persistence(b):.3f}   "
          f"关学习 {persistence(b0):.3f}")
    r_on, r_off = recall_test(b), recall_test(b0)
    print(f"  联想召回（先 A 后 B，B 码里含 A 单元的比例）："
          f"开学习 {r_on['cold']:.3f}→{r_on['cued']:.3f}（+{r_on['lift']:.3f}）   "
          f"关学习 {r_off['cold']:.3f}→{r_off['cued']:.3f}（+{r_off['lift']:.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
