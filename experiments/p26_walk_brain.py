# -*- coding: utf-8 -*-
"""走路题：脑自己输出 12 条肌肉的力度，看机身有没有前进。

判据：位移 > 0.30 且不摔。位移来自 `body.data.qpos[0]`。

用法（需要 mujoco，用系统 Python）：
    python experiments/p26_walk_brain.py --ticks 150
    python experiments/p26_walk_brain.py --k_near 4 --region_links 5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec                     # noqa: E402
from agi.body_go2 import Go2Body                      # noqa: E402
from agi.encoding import Observation                  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=150)
    p.add_argument("--preset", default="full")
    p.add_argument("--offset", type=float, default=0.12)
    p.add_argument("--k_near", type=int, default=0)
    p.add_argument("--region_links", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    sp = build_spec(args.preset, seed=7 + args.seed)
    if args.k_near:
        sp.k_near = args.k_near
    if args.region_links >= 0:
        sp.region_links = args.region_links
    brain = build(sp, verbose=False)
    eng = brain.engine

    body = Go2Body(posture_offset=args.offset)
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])

    px = sp.px
    obs = Observation(vision=np.full((px, px, 3), 0.5, np.float32),
                      audio=np.full(sp.audio_bands * sp.audio_frames, 0.34, np.float32),
                      touch=np.full(sp.touch_ch, 0.34, np.float32),
                      proprio=np.full(sp.proprio_ch, 0.34, np.float32),
                      balance=np.full(sp.balance_ch, 0.34, np.float32))
    xs = []
    for t in range(args.ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(obs))
        eng.step()
        body.step(brain.forces())
        xs.append(float(body.data.qpos[0]))
    dx = xs[-1] - x0
    ok = (not body.fallen()) and abs(dx) > 0.30
    print(f"位移 {dx:+.3f} 摔倒 {body.fallen()} 末高 {body.state()['height']:.3f}")
    print(f"会走 {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
