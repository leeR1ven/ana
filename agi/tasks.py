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
