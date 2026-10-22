# -*- coding: utf-8 -*-
"""任务类题库：真的去做一件事（和"诊断题"分开）。

第一个任务：**注视红球**。
  · 世界：一张图上有一个红球（随机位置）；
  · 动作：注视窗中心 (gx, gy) ∈ [-1, 1]²；
  · 目标：在 N 拍内把注视窗移到球心附近（误差 < tol）。

这里只提供"世界 + 判分"，策略由外部给（脑、随机、控制器都行）。
这样任务本身可以独立验证：随机策略应该接近 0 分，比例控制器应该接近满分。
"""
from __future__ import annotations

import numpy as np


class GazeWorld:
    def __init__(self, seed: int = 0, tol: float = 0.12, steps: int = 20):
        self.rng = np.random.default_rng(seed)
        self.tol = float(tol)
        self.steps = int(steps)
        self.ball = np.zeros(2, np.float32)
        self.gaze = np.zeros(2, np.float32)
        self.t = 0

    def reset(self) -> np.ndarray:
        self.ball = self.rng.uniform(-0.8, 0.8, 2).astype(np.float32)
        self.gaze = np.zeros(2, np.float32)
        self.t = 0
        return self.obs()

    def obs(self) -> np.ndarray:
        """观察：球相对注视窗的偏移（这才是"看得见"的东西）。"""
        return (self.ball - self.gaze).astype(np.float32)

    def step(self, action) -> np.ndarray:
        a = np.clip(np.asarray(action, np.float32).reshape(-1), -1, 1)
        if a.size < 2:
            a = np.zeros(2, np.float32)
        self.gaze = np.clip(self.gaze + 0.25 * a[:2], -1, 1)
        self.t += 1
        return self.obs()

    def done(self) -> bool:
        return self.t >= self.steps

    def score(self) -> float:
        return float(np.linalg.norm(self.ball - self.gaze) < self.tol)


def run_episode(policy, seed: int = 0, **kw) -> float:
    w = GazeWorld(seed, **kw)
    o = w.reset()
    while not w.done():
        o = w.step(policy(o))
    return w.score()          # 只看最后有没有对准


def random_policy(o):
    return np.random.uniform(-1, 1, 2)


def controller_policy(o):
    """最简单的手写控制：往球偏移的方向走。"""
    return np.clip(np.asarray(o, np.float32) / 0.25, -1, 1)


if __name__ == "__main__":
    for name, pol in (("随机", random_policy), ("比例控制", controller_policy)):
        scores = [run_episode(pol, seed=s) for s in range(10)]
        print(f"{name}: 成功率 {sum(scores):.0f}/10")

# ------------------------------------------------------------------ 脑策略
class GazeBrain:
    """用脑来做注视：看"球相对注视窗的位置"，输出动作。

    视觉输入是"以注视窗为中心"的画面：球偏右，画面里的球就偏右。
    动作取运动区前两条肌肉的力度（0..1 → -1..1）。
    """

    def __init__(self, seed: int = 0):
        from .spec import build_spec
        from .build import build
        from .loop import Cortex
        from .encoding import Observation
        self.Observation = Observation
        self.sp = build_spec("full", seed=7 + seed)
        self.b = build(self.sp, verbose=False)
        self.ctx = Cortex(self.b)
        self.px = self.sp.px
        self._img = np.zeros((self.px, self.px, 3), np.float32)

    def _see(self, off):
        from .world import _blob
        px = self.px
        x = 0.5 + 0.45 * float(np.clip(off[0], -1, 1))
        y = 0.5 + 0.45 * float(np.clip(off[1], -1, 1))
        img = np.maximum(np.full((px, px, 3), 0.05, np.float32),
                         _blob(px, x, y, 0.18, (0.95, 0.2, 0.1)))
        sp = self.sp
        return self.Observation(
            vision=img, audio=np.zeros((sp.audio_bands, sp.audio_frames), np.float32),
            touch=np.full(sp.touch_ch, 0.5, np.float32),
            proprio=np.full(sp.proprio_ch, 0.5, np.float32),
            balance=np.full(sp.balance_ch, 0.5, np.float32))

    def policy(self, off):
        # 肌肉力度是 0..1（静息为 0），不能直接当有符号位移；
        # 用两条肌肉"拮抗"：一条推、一条拉，差值才是方向。
        self.ctx.tick(self._see(off), record=False)
        f = np.asarray(self.b.forces(), np.float32)
        a = np.array([f[2] - f[3], f[9] - f[10]], np.float32) * 2.0
        return np.clip(a, -1, 1)
