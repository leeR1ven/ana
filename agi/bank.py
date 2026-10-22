# -*- coding: utf-8 -*-
"""固定题库：只评行为、固定可复现。

原则（沿用老库 chain-neuron-architecture 的做法）：
  · 每个模型用**同一套题**，不许换题；
  · 只看得分，不看内部实现；
  · 每题 0/1，按类别报通过率。

用法：
    python -m agi.bank            # 给当前默认配置打分
    python -m agi.bank --seed 3   # 换一个随机种子
"""
from __future__ import annotations

import argparse
import numpy as np

from .spec import build_spec
from .build import build
from .loop import Cortex
from .encoding import Observation
from .dentate import dentate_code
from .consolidate import Consolidator
from .world import _blob


# ------------------------------------------------------------------ 工具
def jac(a, b) -> float:
    return len(a & b) / max(1, len(a | b))


def scene(px, ox=0.5, oy=0.5, seed=None, noise=0.0, occlude=0.0):
    rng = np.random.default_rng(seed)
    if seed is None:
        img = np.maximum(np.zeros((px, px, 3), np.float32) + 0.05,
                         _blob(px, ox, oy, 0.25, (0.95, 0.2, 0.1)))
    else:
        img = rng.random((px, px, 3), dtype=np.float32)
    if occlude > 0:
        k = int(px * occlude)
        img[:k] = 0.05
    if noise > 0:
        img = np.clip(img + rng.normal(0, noise, img.shape).astype(np.float32), 0, 1)
    return img


def obs_of(px, img, sp):
    return Observation(
        vision=np.asarray(img, np.float32),
        audio=np.zeros((sp.audio_bands, sp.audio_frames), np.float32),
        touch=np.full(sp.touch_ch, 0.5, np.float32),
        proprio=np.full(sp.proprio_ch, 0.5, np.float32),
        balance=np.full(sp.balance_ch, 0.5, np.float32))


class Harness:
    def __init__(self, seed: int = 0, **kw):
        self.sp = build_spec("full", seed=7 + seed, **kw)
        self.b = build(self.sp, verbose=False)
        self.eng = self.b.engine
        self.ctx = Cortex(self.b)
        self.dg = np.asarray(self.b.ids["dentate"])
        self.px = self.sp.px

    def run(self, img, n=12, reset=True):
        o = obs_of(self.px, img, self.sp)
        if reset:
            self.eng.act[:] = 0.0
            self.eng.reset_adaptation()
        for _ in range(n):
            self.ctx.tick(o, record=False)

    def pfc(self):
        return set(np.asarray(self.b.pfc_code()).tolist())

    def dent(self):
        return set(dentate_code(self.eng, self.dg).tolist())

    def cue(self, ids, n=8):
        self.eng.act[:] = 0.0
        self.eng.reset_adaptation()
        ids = list(ids)
        for _ in range(n):
            self.eng.clear_external()
            self.eng.set_input(np.zeros(self.b.encoder.n_in, np.float32))
            self.eng.set_external(ids, np.ones(len(ids), np.float32))
            self.eng.step()
        return np.asarray(self.eng.drive()).copy()


# ------------------------------------------------------------------ 题目
# 两类题，意义不同：
#   诊断类：检查架构有没有问题（同图同码、稀疏、噪声…），不求"做事"
#   任务类：真的去做一件事（注视红球、站住、走过去…），评的是能力
CASES = []


def case(cat, name, fresh=True):
    def deco(fn):
        CASES.append((cat, name, fn, fresh))
        return fn
    return deco


@case("感知", "同图同码")
def _(h):
    h.run(scene(h.px, seed=None))
    a = h.pfc()
    h.run(scene(h.px, seed=None))
    return a == h.pfc()


@case("感知", "近图相似")
def _(h):
    h.run(scene(h.px, 0.50, 0.50))
    a = h.pfc()
    h.run(scene(h.px, 0.54, 0.50))
    return jac(a, h.pfc()) > 0.6


@case("感知", "远图分离")
def _(h):
    h.run(scene(h.px, 0.50, 0.50))
    a = h.pfc()
    h.run(scene(h.px, seed=9))
    return jac(a, h.pfc()) < 0.4


@case("感知", "稀疏活跃")
def _(h):
    h.run(scene(h.px))
    frac = len(h.pfc()) / max(1, len(h.b.pfc_ids))
    return 0.01 < frac < 0.15


@case("抗干扰", "噪声不变")
def _(h):
    h.run(scene(h.px, 0.5, 0.5))
    a = h.pfc()
    h.run(scene(h.px, 0.5, 0.5, noise=0.03))
    return jac(a, h.pfc()) > 0.5


@case("抗干扰", "遮挡一半")
def _(h):
    h.run(scene(h.px, 0.4, 0.4))
    a = h.pfc()
    h.run(scene(h.px, 0.4, 0.4, occlude=0.5))
    return jac(a, h.pfc()) > 0.3


@case("抗干扰", "少量位移")
def _(h):
    h.run(scene(h.px, 0.40, 0.40))
    a = h.pfc()
    h.run(scene(h.px, 0.45, 0.45))
    return jac(a, h.pfc()) > 0.4


@case("记忆", "即时回忆")
def _(h):
    # 要跟"第 8 拍当时的内容"比，而不是最后一拍（码每拍会漂一点）
    snaps = []
    for k in range(14):
        h.run(scene(h.px, 0.5, 0.5), n=1, reset=(k == 0))   # 只在第一拍清零，否则永远settle不下来
        snaps.append(h.dent())
    snap = snaps[8]
    h.b.binder.flush()                  # 时间绑定是 binder 写的（组 64），不是 consolidator
    slot = int(h.b.time_ring.ids[8])
    h.eng.act[:] = 0.0
    h.eng.reset_adaptation()
    for _ in range(8):
        h.eng.clear_external()
        h.eng.set_input(np.zeros(h.b.encoder.n_in, np.float32))
        h.eng.set_external([slot], np.ones(1, np.float32))
        h.eng.step()
    got = set(dentate_code(h.eng, h.dg).tolist())
    return len(got & snap) / max(1, len(snap)) > 0.3


@case("记忆", "长期保持")
def _(h):
    co = Consolidator(h.eng, lr=1.0, cap=32)   # 独立：只喂"稳定后的码"
    A_img = scene(h.px, 0.5, 0.5)
    B_img = scene(h.px, seed=4)
    for _ in range(8):
        h.run(A_img, n=14)
        a = h.dent()
        h.run(B_img, n=14)
        co.step(list(h.dent()))
    co.flush()
    A = set(a)
    eng = h.eng
    for _ in range(600):
        h.ctx.tick(obs_of(h.px, scene(h.px, seed=11), h.sp), record=False)
    d = h.cue(A)
    rng = np.random.default_rng(0)
    rnd = rng.choice(h.dg, 400, replace=False)
    return float(np.mean(d[list(A)])) > 3 * float(np.mean(d[rnd]))


@case("记忆", "多段顺序")
def _(h):
    # 三个画面的两两重合都 ≤0.11（实测），这题才公平
    imgs = {1: scene(h.px, seed=4), 2: scene(h.px, 0.5, 0.5), 3: scene(h.px, seed=9)}
    co = Consolidator(h.eng, lr=1.0, cap=32)
    codes = {}
    for _ in range(10):
        for x in (1, 2, 3, 2):
            h.run(imgs[x], n=14)
            codes[x] = h.dent()
            co.step(list(h.dent()))
    co.flush()
    d = h.cue(codes[1], n=1)          # 只走一拍：看"直接联想"，两拍就会经由 B 传到 C
    fwd = float(np.mean(d[list(codes[2])]))
    skip = float(np.mean(d[list(codes[3])]))
    return fwd > skip


@case("感知", "颜色辨别")
def _(h):
    red = scene(h.px, 0.5, 0.5)
    green = np.maximum(np.zeros((h.px, h.px, 3), np.float32) + 0.05,
                       _blob(h.px, 0.5, 0.5, 0.25, (0.1, 0.9, 0.2)))
    h.run(red)
    a = h.pfc()
    h.run(green)
    return jac(a, h.pfc()) < 0.4


@case("感知", "尺度辨别")
def _(h):
    big = np.maximum(np.zeros((h.px, h.px, 3), np.float32) + 0.05,
                     _blob(h.px, 0.5, 0.5, 0.30, (0.95, 0.2, 0.1)))
    small = np.maximum(np.zeros((h.px, h.px, 3), np.float32) + 0.05,
                       _blob(h.px, 0.5, 0.5, 0.10, (0.95, 0.2, 0.1)))
    h.run(big)
    a = h.pfc()
    h.run(small)
    return jac(a, h.pfc()) < 0.6


@case("抗干扰", "整体亮度")
def _(h):
    base = scene(h.px, 0.45, 0.45)
    h.run(base)
    a = h.pfc()
    h.run(np.clip(base + 0.10, 0, 1).astype(np.float32))
    return jac(a, h.pfc()) > 0.4


@case("记忆", "两段时间分辨")
def _(h):
    A = scene(h.px, 0.30, 0.30)
    B = scene(h.px, seed=7)
    for k in range(8):
        h.run(A, n=1, reset=(k == 0))
    h.b.binder.flush()
    codeA = h.dent()
    for k in range(8, 16):
        h.run(B, n=1, reset=False)
    codeB = h.dent()
    slotA = int(h.b.time_ring.ids[5])
    h.eng.act[:] = 0.0
    h.eng.reset_adaptation()
    for _ in range(8):
        h.eng.clear_external()
        h.eng.set_input(np.zeros(h.b.encoder.n_in, np.float32))
        h.eng.set_external([slotA], np.ones(1, np.float32))
        h.eng.step()
    got = set(dentate_code(h.eng, h.dg).tolist())
    ha = len(got & codeA) / max(1, len(codeA))
    hb = len(got & codeB) / max(1, len(codeB))
    return ha > 0.2 and ha > hb


def _rules(h):
    return None


def run(seed: int = 0, only: str | None = None):
    print(f"题库评测 seed={seed}")
    total = {True: 0, False: 0}
    by_cat = {}
    for cat, name, fn, fresh in CASES:
        if only and only not in cat:
            continue
        try:
            ok = bool(fn(Harness(seed))) if fresh else bool(fn(h))
        except Exception as e:                      # 出题本身错了也要算失败
            ok = False
            name = f"{name} (错误: {type(e).__name__})"
        if not fresh:
            pass
        total[ok] += 1
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][0] += int(ok)
        by_cat[cat][1] += 1
        print(f"  [{'通过' if ok else '失败'}] {cat} · {name}")
    print()
    for cat, (k, n) in by_cat.items():
        kind = "诊断" if cat in ("感知", "抗干扰", "记忆") else "任务"
        print(f"  [{kind}] {cat}: {k}/{n} = {k / max(1, n):.0%}")
    k, n = total[True], total[True] + total[False]
    print(f"  总计: {k}/{n} = {k / max(1, n):.0%}")
    return k, n


if False:  # 入口已移到文件末尾（新题在它之后定义，不能提前跑）
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    run(a.seed, a.only)

@case("感知", "跨区一致")
def _(h):
    # 同一张图两次，视觉 L5 的码必须完全一样
    ids = np.concatenate([L.ids for n, L in h.b.layers.items() if n.endswith(".L5")])
    def l5():
        a = np.asarray(h.eng.act).copy()
        return set(np.flatnonzero(a[ids] > 0.5).tolist())
    h.run(scene(h.px, 0.5, 0.5))
    a = l5()
    h.run(scene(h.px, 0.5, 0.5))
    return a == l5()


@case("感知", "位置分辨")
def _(h):
    h.run(scene(h.px, 0.25, 0.25))
    a = h.pfc()
    h.run(scene(h.px, 0.75, 0.75))
    return jac(a, h.pfc()) < 0.7


@case("记忆", "短期覆盖")
def _(h):
    A = scene(h.px, 0.3, 0.3)
    B = scene(h.px, seed=7)
    for k in range(10):
        h.run(A, n=1, reset=(k == 0))
    slot = int(h.b.time_ring.ids[5])
    codeA = h.dent()
    h.b.binder.flush()
    # 走满一圈相关位置（时间格被重用）
    for k in range(600):
        h.ctx.tick(obs_of(h.px, B if k % 20 else scene(h.px, 0.7, 0.7), h.sp), record=False)
    h.b.binder.flush()
    codeB = h.dent()
    h.eng.act[:] = 0.0
    h.eng.reset_adaptation()
    for _ in range(8):
        h.eng.clear_external()
        h.eng.set_input(np.zeros(h.b.encoder.n_in, np.float32))
        h.eng.set_external([slot], np.ones(1, np.float32))
        h.eng.step()
    got = set(dentate_code(h.eng, h.dg).tolist())
    ha = len(got & codeA) / max(1, len(codeA))
    hb = len(got & codeB) / max(1, len(codeB))
    return hb >= ha


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    run(a.seed, a.only)
# ------------------------------------------------------------------ 外部题
# 运动、规则类需要 mujoco（只在系统 Python 里有），所以走子进程，不混进主环境。
_EXTERNAL = (
    ("运动", "站立不摔", "experiments/p5_stand.py --ticks 80", "摔倒 False"),
    ("规则", "名字→动作", "experiments/p14_rule_table.py", "规则表可用"),
)


def run_external(py=None):
    import subprocess
    py = py or os.environ.get("ANA_PY", "")
    ok = bad = 0
    for cat, name, script, must in _EXTERNAL:
        if not py:
            print(f"  [跳过] {cat} · {name}（未设置 ANA_PY）")
            continue
        try:
            r = subprocess.run([py, "-X", "utf8"] + script.split(),
                               capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace",
                               cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            hit = must in (r.stdout or "")
        except Exception:
            hit = False
        ok += hit
        bad += (not hit)
        print(f"  [{'通过' if hit else '失败'}] {cat} · {name}")
    return ok, bad
