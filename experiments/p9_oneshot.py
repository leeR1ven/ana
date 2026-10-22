# -*- coding: utf-8 -*-
"""P9：融合——一次学会（反向赫布），学过不忘。

做法：
  1. 让网络看一个画面 A，记下它亮的前额叶单元；
  2. **一次**把它们连到"站起来"的力度档上（权重直接给满，不再训练）；
  3. 之后画面全黑、只点 A：看力度是不是 0.5（站），狗能不能从趴着站起来；
  4. 中间穿插看别的画面（B/C）200 拍，再回来点 A——力度还在，就是"学过不忘"。

    python experiments/p9_oneshot.py --ticks 120
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec                                        # noqa: E402
from agi.body_go2 import Go2Body                                        # noqa: E402
from experiments.p1_recall import run_obs, scene_obs                    # noqa: E402

G_ONESHOT = 60


def teach_once(eng, src_ids, force_cells, levels: int = 5, w: float = 6.0) -> int:
    """一次学会：把 src_ids 接到每块肌肉的前 levels 档力度细胞上。"""
    src, dst = [], []
    for m in range(len(force_cells)):
        f = force_cells[m]
        src.append(np.full(levels, src_ids[0], np.int32) if False else
                   np.repeat(src_ids, levels))
        dst.append(np.tile(f[:levels], len(src_ids)))
    eng.add_edges(np.concatenate(src), np.concatenate(dst),
                  np.full(sum(len(s) for s in src), w, np.float32), group=G_ONESHOT)
    eng.rebuild_backend()
    return int(sum(len(s) for s in src))


def think(eng, brain, cue_ids, ticks: int = 12):
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(np.zeros(brain.encoder.n_in, np.float32))
        if cue_ids is not None:
            eng.set_external(cue_ids, np.ones(len(cue_ids), np.float32))
        eng.step()
    return brain.forces().copy()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="full")
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--offset", type=float, default=0.12)
    args = p.parse_args(argv)

    sp = build_spec(args.preset)
    brain = build(sp, verbose=False)
    eng = brain.engine

    run_obs(brain, scene_obs(sp, 1))
    A = brain.pfc_code()
    n_edges = teach_once(eng, A, brain.motor["force"])
    print(f"一次学会：源 {len(A)} 个前额叶单元 → 12 块肌肉 × 5 档，共 {n_edges} 条边")

    f_cue = think(eng, brain, list(A))
    print(f"只点 A：力度均值 {f_cue.mean():.2f}（期望 0.50）")

    # 站：从趴着开始，边点 A 边跑身体
    body = Go2Body(posture_offset=args.offset)
    body.reset()
    body.data.qpos[2] = 0.10
    body.data.qpos[7:19] = np.array([0.0, 1.6, -2.6] * 4)
    body.mj.mj_forward(body.model, body.data)
    h0 = body.state()["height"]
    for _ in range(args.ticks):
        f = think(eng, brain, list(A), ticks=1)
        body.step(f)
    h1 = body.state()["height"]
    print(f"身体：{h0:.3f} → {h1:.3f}，摔倒 {body.fallen()}")

    # 打断：看别的画面 200 拍（不学习），再回来
    for t in range(200):
        run_obs(brain, scene_obs(sp, 2 if t % 2 else 3), ticks=1)
    run_obs(brain, scene_obs(sp, 2))
    f_after = think(eng, brain, list(A))
    print(f"看过别的东西之后，再点 A：力度均值 {f_after.mean():.2f}（应仍是 0.50）")
    kept = abs(f_after.mean() - f_cue.mean()) < 0.02
    print(f"结论：{'学过不忘（一次学会成立）' if kept else '记忆被冲掉了'}"
          f"（教完 {f_cue.mean():.3f}，打断后 {f_after.mean():.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
