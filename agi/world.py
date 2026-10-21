# -*- coding: utf-8 -*-
"""虚拟世界 + 虚拟身体（v0 用；接宇树 Go2 时换掉这一层即可）。

这里不做真物理：只做"动作会改变下一拍感觉"的那部分因果，用来验证
感知→链→前额叶→运动区→肌肉→身体→新感觉这条闭环是通的。

`sample_observation()` 给标定用：随机图 / 平滑梯度 / 色块，加上音调、
触地、本体、平衡。标定必须用"有结构的输入"，不能用纯噪声——纯噪声会把
前额叶门槛标到只有噪声能过。
"""
from __future__ import annotations

import numpy as np

from .encoding import Observation


# ---------------------------------------------------------------- 合成场景
def _blob(px: int, cx: float, cy: float, r: float, color) -> np.ndarray:
    yy, xx = np.mgrid[0:px, 0:px].astype(np.float32) / max(1, px - 1)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    m = np.exp(-(d / max(1e-3, r)) ** 2)[..., None]
    img = np.zeros((px, px, 3), np.float32) + 0.05
    return np.clip(img + m * np.asarray(color, np.float32), 0, 1)


def sample_observation(spec, rng: np.random.Generator, tick: int = 0) -> Observation:
    """有结构的随机场景：随机底色 + 1~2 个色块 + 平滑梯度；偶尔纯噪声。"""
    px = spec.px
    kind = int(rng.integers(0, 4))
    if kind == 0:
        img = rng.random((px, px, 3), dtype=np.float32)
    elif kind == 1:
        g = np.linspace(0, 1, px, dtype=np.float32)
        img = np.stack([np.tile(g, (px, 1)), np.tile(g[:, None], (1, px)),
                        np.full((px, px), 0.5, np.float32)], -1)
        img = np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1).astype(np.float32)
    else:
        img = np.zeros((px, px, 3), np.float32) + rng.uniform(0, 0.3)
        for _ in range(int(rng.integers(1, 3))):
            img = np.maximum(img, _blob(px, rng.uniform(0.15, 0.85),
                                        rng.uniform(0.15, 0.85),
                                        rng.uniform(0.08, 0.28),
                                        rng.uniform(0.2, 1.0, 3)))
    bands, frames = spec.audio_bands, spec.audio_frames
    au = np.zeros((bands, frames), np.float32)
    for _ in range(int(rng.integers(1, 3))):
        f = int(rng.integers(0, bands))
        au[max(0, f - 1):f + 2, :] += rng.uniform(0.5, 1.0)
    au = np.clip(au + rng.normal(0, 0.05, au.shape), 0, 1).astype(np.float32)
    touch = np.clip(rng.random(spec.touch_ch, dtype=np.float32) * 0.4, 0, 1)
    proprio = np.clip(rng.random(spec.proprio_ch, dtype=np.float32), 0, 1)
    balance = np.clip(0.5 + rng.normal(0, 0.12, spec.balance_ch), 0, 1).astype(np.float32)
    return Observation(vision=img, audio=au, touch=touch, proprio=proprio,
                       balance=balance, tick=tick)


# ---------------------------------------------------------------- 虚拟身体
class VirtualDog:
    """极简平面运动学：前/后腿力差→前进，左/右腿力差→转向，总力→站高。

    坐标：x 前、y 左、θ 朝向。目标是一个红块；"看见"的目标位置决定画面，
    声音的音高 ∝ 距离。只求闭环可观察，不求物理真实。
    """

    def __init__(self, spec, seed: int = 0):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.height = 0.22
        self.speed = 0.0
        self.target = np.array([1.2, 0.6], np.float32)
        self.t = 0
        self.last_forces = np.zeros(spec.n_muscle, np.float32)

    # -- 腿的编号：0..2 FR / 3..5 FL / 6..8 RR / 9..11 RL（与 Go2 一致）
    def _leg_means(self, f):
        return (float(np.mean(f[0:3])), float(np.mean(f[3:6])),
                float(np.mean(f[6:9])), float(np.mean(f[9:12])))

    def step(self, forces, dt: float = 0.02):
        f = np.asarray(forces, np.float32)
        self.last_forces = f
        fr, fl, rr, rl = self._leg_means(f)
        front, back = (fr + fl) / 2, (rr + rl) / 2
        left, right = (fl + rl) / 2, (fr + rr) / 2
        drive = front - back
        self.speed = float(np.clip(0.8 * self.speed + 1.5 * drive, -0.6, 0.8))
        turn = float(np.clip((left - right) * 1.2, -1.0, 1.0))
        self.th += turn * dt
        self.x += np.cos(self.th) * self.speed * dt
        self.y += np.sin(self.th) * self.speed * dt
        total = float(np.mean(f))
        self.height = float(np.clip(0.18 + 0.25 * total, 0.12, 0.26))
        self.t += 1
        return self

    # -- 感知
    def observe(self) -> Observation:
        sp = self.spec
        px = sp.px
        d = self.target - np.array([self.x, self.y])
        dist = float(np.hypot(*d))
        rel = self._wrap(np.arctan2(d[1], d[0]) - self.th)      # -pi..pi
        # 画面：目标在视野里的水平位置由 rel 决定；越远越小越暗
        cx = float(np.clip(0.5 + rel / 1.2, -0.2, 1.2))
        size = float(np.clip(0.10 + 0.06 * (1.0 / (0.5 + dist)), 0.06, 0.3))
        img = np.zeros((px, px, 3), np.float32) + 0.04
        img = np.maximum(img, _blob(px, cx, 0.55, size, (0.9, 0.15, 0.1)))
        # 地平线：上半暗、下半亮
        img[: px // 2] *= 0.5
        img += self.rng.normal(0, 0.01, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)
        # 声音：距离越近音越高（频带位置）
        au = np.zeros((sp.audio_bands, sp.audio_frames), np.float32)
        band = int(np.clip((1.0 - dist / 3.0) * sp.audio_bands, 0, sp.audio_bands - 1))
        au[max(0, band - 1):band + 2, :] = 0.9
        au = np.clip(au + self.rng.normal(0, 0.03, au.shape), 0, 1)
        # 触地：站得越低、力越大，触地越强
        touch = np.zeros(sp.touch_ch, np.float32)
        contact = float(np.clip(0.9 - (self.height - 0.12) * 3.0, 0.0, 1.0))
        touch[: min(12, sp.touch_ch)] = contact
        if sp.touch_ch > 12:
            touch[12:] = 0.1
        # 本体：关节角/角速度/力矩的粗糙代理
        proprio = np.zeros(sp.proprio_ch, np.float32)
        n12 = min(12, sp.proprio_ch)
        proprio[:n12] = np.clip(self.last_forces[:n12] * 0.8 + 0.3, 0, 1)
        if sp.proprio_ch > 12:
            proprio[12:24] = 0.5
        if sp.proprio_ch > 24:
            proprio[24:36] = np.clip(self.last_forces[: min(12, sp.proprio_ch - 24)], 0, 1)
        # 平衡：滚转/俯仰由左右、前后力差代理；陀螺≈转向
        balance = np.zeros(sp.balance_ch, np.float32)
        fr, fl, rr, rl = self._leg_means(self.last_forces)
        vals = [0.5 + (fl + rl - fr - rr) * 0.4,
                0.5 + (fr + fl - rr - rl) * 0.4,
                0.5 + self.th / 3.0,
                0.5 + float(np.clip(self.speed, -1, 1)) * 0.4,
                0.5, 0.5 + np.sin(self.t * 0.1) * 0.1,
                float(np.clip(self.height / 0.3, 0, 1)), 0.5, 0.5, 0.5]
        balance[: min(len(vals), sp.balance_ch)] = vals[: sp.balance_ch]
        balance = np.clip(balance, 0, 1).astype(np.float32)
        return Observation(vision=img, audio=au, touch=touch, proprio=proprio,
                           balance=balance, tick=self.t,
                           meta=dict(dist=dist, rel=rel, speed=self.speed))

    @staticmethod
    def _wrap(a: float) -> float:
        while a > np.pi:
            a -= 2 * np.pi
        while a < -np.pi:
            a += 2 * np.pi
        return float(a)
