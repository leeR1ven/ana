# -*- coding: utf-8 -*-
"""P3：让大脑真的驱动身体——"看到哪边就往哪边倾"。

做法：
  1. 示范：色块在左 → 左腿力度高、右腿低（会往左倾）；在右则相反；
  2. 抄成连接：视觉区的神经元 → 12 块肌肉的力度档（log-odds 权重 + 门槛，group 42）；
  3. 接上 MuJoCo Go2，只给画面、不给别的，看狗的实际倾角跟画面方向是否一致；
  4. 对照：同样的画面，但把编译出来的连接切断（力度当 0.5 平），倾角应该不随画面变。

    python experiments/p3_cue_body.py --ticks 80
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
from agi.encoding import Observation                                    # noqa: E402
from agi.loop import Cortex                                             # noqa: E402
from agi.world import _blob                                             # noqa: E402
from experiments.p2_compile import compile_cells, install, G_STATE2FORCE  # noqa: E402

LEFT_LEGS = np.array([3, 4, 5, 9, 10, 11])
RIGHT_LEGS = np.array([0, 1, 2, 6, 7, 8])


def cue_obs(spec, x: float) -> Observation:
    """色块在画面里的水平位置 x（0=最左，1=最右）。"""
    img = np.zeros((spec.px, spec.px, 3), np.float32) + 0.05
    img = np.maximum(img, _blob(spec.px, float(np.clip(x, 0.08, 0.92)), 0.5, 0.2,
                                (0.95, 0.35, 0.1)))
    return Observation(vision=img,
                       audio=np.zeros((spec.audio_bands, spec.audio_frames), np.float32),
                       touch=np.full(spec.touch_ch, 0.3, np.float32),
                       proprio=np.full(spec.proprio_ch, 0.5, np.float32),
                       balance=np.full(spec.balance_ch, 0.5, np.float32))


def teacher_forces(x: float) -> np.ndarray:
    """示范：色块越靠左，左腿越用力（力度 1 表示关节往"撑起来"那侧偏）。"""
    lean = 1.0 - 2.0 * float(np.clip(x, 0.0, 1.0))     # +1 最左，-1 最右
    f = np.full(12, 0.5, np.float32)
    f[LEFT_LEGS] = 0.5 + 0.45 * lean
    f[RIGHT_LEGS] = 0.5 - 0.45 * lean
    return np.clip(f, 0, 1)


def run_state(brain, obs, ticks=16):
    eng = brain.engine
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(obs))
        eng.step()
    return eng.act_np.copy()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="full")
    p.add_argument("--ticks", type=int, default=60)
    p.add_argument("--offset", type=float, default=0.12, help="力度→关节角偏置（弧度）")
    args = p.parse_args(argv)

    sp = build_spec(args.preset, pfc_topk=0.005, pfc_adapt=0.01)
    brain = build(sp, verbose=False)
    eng = brain.engine
    state_ids = np.concatenate([brain.layers[f"vision.L{d}"].ids for d in (3, 4, 5)])
    force_ids = np.concatenate([brain.motor["force"][m] for m in range(sp.n_muscle)])

    # ---- 1. 采示范：色块在 9 个位置
    xs = np.linspace(0.1, 0.9, 9)
    A, D, K = [], [], []
    for x in xs:
        act = run_state(brain, cue_obs(sp, x))
        A.append(act[state_ids])
        D.append(eng.drive().numpy()[force_ids])
        K.append(np.rint(teacher_forces(x) * sp.n_force).astype(int))
    A, D, K = np.stack(A), np.stack(D), np.stack(K)
    W, thr, acc = compile_cells(A, D, K, sp.n_force)
    print(f"编译：{int(np.isfinite(thr).sum())} 个力度细胞，训练集准确率 "
          f"{float(acc[np.isfinite(thr)].mean()):.3f}")
    install(brain, state_ids, force_ids, W, thr)
    ctx = Cortex(brain, trace=False)

    # ---- 2. 接狗跑：左、中、右三个画面各跑一段，看倾角
    def run_body(x, ticks, use_brain=True):
        body = Go2Body(posture_offset=args.offset)
        body.reset()
        rolls = []
        for _ in range(ticks):
            obs = cue_obs(sp, x)
            tk = ctx.tick(obs, record=False)
            body.step(tk.forces if use_brain else np.full(12, 0.5, np.float32))
            rolls.append(body.state()["roll"])
        return float(np.mean(rolls[-max(10, ticks // 3):]))

    print("画面位置 → 狗的实际倾角（负=往左倾，正=往右倾）")
    for x in (0.15, 0.5, 0.85):
        r_on = run_body(x, args.ticks, True)
        print(f"  x={x:.2f}：大脑驱动时 roll {r_on:+.4f}")
    print("对照：同样画面，力度固定 0.5（大脑不参与）")
    for x in (0.15, 0.85):
        r_off = run_body(x, args.ticks, None if False else False)
        print(f"  x={x:.2f}：roll {r_off:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
