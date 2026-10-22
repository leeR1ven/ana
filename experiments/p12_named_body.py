# -*- coding: utf-8 -*-
"""P12：名字 → 规则 → 运动区 → 身体（整条链）。

  1. 标定"站"这个画面；
  2. 用名字写规则：站 → 12 块肌肉各 5 档；
  3. 画面全黑、只点"站"：狗从趴着站起来。

这就是"用名字操控身体"的最小闭环。

    python experiments/p12_named_body.py --ticks 120
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec                                     # noqa: E402
from agi.body_go2 import Go2Body                                      # noqa: E402
from agi.naming import NameTable                                      # noqa: E402
from experiments.p1_recall import scene_obs                           # noqa: E402

TMP = os.path.join(ROOT, "_names_body.json")
GROUP = 61


def install(eng, brain, nt: NameTable, name: str, levels: int, w: float = 6.0) -> int:
    src = nt.get(name)
    total = 0
    for m in range(brain.spec.n_muscle):
        dst = brain.motor["force"][m][:levels]
        s = np.repeat(src, len(dst))
        d = np.tile(dst, len(src))
        eng.add_edges(s.astype(np.int32), d.astype(np.int32),
                      np.full(len(s), w, np.float32), group=GROUP)
        total += len(s)
    eng.rebuild_backend()
    return total


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--offset", type=float, default=0.12)
    args = p.parse_args(argv)

    sp = build_spec("full")
    brain = build(sp, verbose=False)
    nt = NameTable(TMP)
    nt.calibrate(brain, scene_obs(sp, 1), "站")
    nt.save(TMP)
    n = install(brain.engine, brain, nt, "站", 5)
    print(f"名字「站」：{len(nt.get('站'))} 个单元 → {n} 条边")

    eng = brain.engine
    body = Go2Body(posture_offset=args.offset)
    body.reset()
    body.data.qpos[2] = 0.10
    body.data.qpos[7:19] = np.array([0.0, 1.6, -2.6] * 4)
    body.mj.mj_forward(body.model, body.data)
    print(f"起点高度 {body.state()['height']:.3f}（趴着）")

    for _ in range(args.ticks):
        eng.act[:] = 0.0 if _ == 0 else eng.act
        eng.clear_external()
        eng.set_input(np.zeros(brain.encoder.n_in, np.float32))
        eng.set_external(nt.get("站"), np.ones(len(nt.get("站")), np.float32))
        eng.step()
        body.step(brain.forces())
    s = body.state()
    print(f"点「站」{args.ticks} 拍后：高度 {s['height']:.3f}  倾角 "
          f"{s['roll']:+.3f}/{s['pitch']:+.3f}  摔倒 {body.fallen()}")
    ok = (not body.fallen()) and s["height"] > 0.23
    print(f"结论：{'用名字站起来了' if ok else '没站起来'}")
    os.remove(TMP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
