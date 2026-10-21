# -*- coding: utf-8 -*-
"""地基体检：只用架构原话里的几条，检查现在的网络做不做得到。

不测"能力"，只测"地基对不对"：
  1. 同一张图看两次，亮的神经元是不是完全一样？
  2. 改动一点点，亮的是不是"相似但不同"（重叠多）？
  3. 换成完全不同的图，重叠是不是明显变少？
  4. 每一层每拍亮几个，是不是恒定？
  5. 五个区是不是都活着、都送进了前额叶？
  6. 前额叶内部是不是真的没有连接？

输出全是大白话，不带任何我自造的术语。

    python experiments/foundation_check.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec                                       # noqa: E402
from agi.build import G                                                 # noqa: E402
from agi.encoding import Observation                                    # noqa: E402
from agi.world import _blob                                             # noqa: E402


def scene(spec, cx=0.5, cy=0.5, color=(0.9, 0.2, 0.1), r=0.18, noise=0.0, seed=None):
    if seed is not None:                 # 整张图都不一样（公平的"换一个"对照）
        img = np.random.default_rng(seed).random((spec.px, spec.px, 3)).astype(np.float32)
    else:
        img = np.zeros((spec.px, spec.px, 3), np.float32) + 0.05
        img = np.maximum(img, _blob(spec.px, cx, cy, r, color))
    if noise:
        img = np.clip(img + np.random.default_rng(0).normal(0, noise, img.shape), 0, 1)
    au = np.zeros((spec.audio_bands, spec.audio_frames), np.float32)
    band = int(np.clip(cx * spec.audio_bands, 0, spec.audio_bands - 1))
    au[band:band + 2] = 0.9
    return Observation(vision=img.astype(np.float32), audio=au,
                       touch=np.full(spec.touch_ch, 0.3, np.float32),
                       proprio=np.full(spec.proprio_ch, 0.5, np.float32),
                       balance=np.full(spec.balance_ch, 0.5, np.float32))


def code(brain, obs, ticks: int = 18) -> np.ndarray:
    """把网络清零、喂同一张图 ticks 拍，返回前额叶亮着的编号。"""
    brain.engine.act[:] = 0.0
    brain.engine.reset_adaptation()
    for _ in range(ticks):
        brain.engine.clear_external()
        brain.engine.set_input(brain.encoder.encode(obs))
        brain.engine.step()
    return brain.pfc_code()


def jacc(a, b) -> float:
    A, B = set(a.tolist()), set(b.tolist())
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))


def main() -> int:
    sp = build_spec("full")
    brain = build(sp, verbose=False)
    print(f"规模：{brain.engine.n:,} 个神经元、{brain.engine.n_edges:,} 条连接；"
          f"前额叶 {len(brain.pfc_ids):,} 个单元")
    print()

    base = scene(sp, 0.5, 0.5)
    same = code(brain, scene(sp, 0.5, 0.5))
    same2 = code(brain, scene(sp, 0.5, 0.5))
    near = code(brain, scene(sp, 0.54, 0.5))          # 挪动一点点
    far = code(brain, scene(sp, seed=7))              # 整张图都不一样
    print("一、同一张图看两次")
    print(f"   两次亮的前额叶单元完全相同？ {'是' if np.array_equal(np.sort(same), np.sort(same2)) else '否'}"
          f"（{len(same)} 个单元，重合度 {jacc(same, same2):.2f}）")
    print("二、只挪动一点点")
    print(f"   重合度 {jacc(same, near):.2f}（期望高：相似但不是同一个）")
    print("三、换成完全不同的东西")
    print(f"   重合度 {jacc(same, far):.2f}（期望低）")
    print()

    # 4. 每层亮几个是否恒定
    print("四、每一层每拍亮几个（连跑 8 拍，看是不是恒定）")
    stable = True
    for name, L in list(brain.layers.items())[:5]:
        counts = []
        brain.engine.act[:] = 0.0
        for t in range(8):
            brain.engine.set_input(brain.encoder.encode(base))
            brain.engine.step()
            counts.append(int(brain.engine.selected[L.ids].sum()))
        ok = len(set(counts)) == 1 and counts[0] == L.k
        stable &= ok
        print(f"   {name:10s} 亮 {counts[0]:5d} 个（设定 {L.k:5d}）  {'恒定' if ok else '不恒定 ' + str(counts)}")
    print()

    # 5. 五个区是不是都活着、都送进前额叶
    print("五、五个感觉区是否都活着、都送进前额叶")
    brain.engine.act[:] = 0.0
    for _ in range(18):
        brain.engine.set_input(brain.encoder.encode(base))
        brain.engine.step()
    eng = brain.engine
    for r in sp.active_regions():
        l5 = brain.layers[f"{r}.L5"]
        n5 = int((eng.act_np[l5.ids] > 0.35).sum())
        pfc = brain.pfc_groups[(r, 5)]
        bad = brain.filtered.get(f"{r}.L5") if brain.filtered else None
        keep = (~bad) if bad is not None else np.ones(len(pfc), bool)
        np_ = int((eng.act_np[pfc[keep]] > 0.35).sum())
        print(f"   {r:9s} 最后一层亮 {n5:4d} 个；它送进前额叶的单元亮 {np_:4d} 个")
    print()

    # 6. 前额叶内部连接
    n_pl = int((eng.group == G.PL_PFC).sum())
    n_any = int((eng.group == G.PLASTIC).sum())
    print("六、前额叶内部")
    print(f"   前额叶之间的连接：{n_pl} 条（按你的要求必须是 0）")
    print(f"   任何学出来的连接：{n_any} 条（这一版必须也是 0）")

    ok = (np.array_equal(np.sort(same), np.sort(same2)) and jacc(same, near) > 0.6
          and jacc(same, far) < 0.5 and stable and n_pl == 0 and n_any == 0)
    print("\n结论：" + ("地基符合" if ok else "有不符合的地方，见上面"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
