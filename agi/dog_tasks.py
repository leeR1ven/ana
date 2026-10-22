# -*- coding: utf-8 -*-
"""狗技能题：每项一个判据，跑一次给一个分（0/1 + 指标）。

需要 mujoco（系统 Python）。用法：
    python -m agi.dog_tasks --skill walk
    python -c "from agi.dog_tasks import run_skill; print(run_skill({}, 'walk'))"
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec          # noqa: E402
from agi.body_go2 import Go2Body           # noqa: E402
from agi.encoding import Observation       # noqa: E402


SKILLS = ("walk0", "walk1", "walk2", "walk3", "walk_to_ball",
          "turn1", "turn2", "getup1", "getup2")


def _brain(kw, preset="small"):
    sp = build_spec(preset, seed=7 + int(kw.get("seed", 0)))
    for k in ("k_near", "region_links", "n_time"):
        if k in kw:
            setattr(sp, k, int(kw[k]))
    for k in ("w_lo", "w_hi"):
        if k in kw:
            setattr(sp, k, float(kw[k]))
    return sp, build(sp, verbose=False)


def _obs(sp, img):
    px = sp.px
    v = np.full((px, px, 3), 0.5, np.float32) if img is None else img
    return Observation(vision=v,
                       audio=np.full(sp.audio_bands * sp.audio_frames, 0.34, np.float32),
                       touch=np.full(sp.touch_ch, 0.34, np.float32),
                       proprio=np.full(sp.proprio_ch, 0.34, np.float32),
                       balance=np.full(sp.balance_ch, 0.34, np.float32))


def _ball(sp, x_norm):
    """把红球画在水平位置 x_norm ∈ [0,1]。"""
    from agi.world import _blob
    px = sp.px
    return np.maximum(np.full((px, px, 3), 0.05, np.float32),
                      _blob(px, float(np.clip(x_norm, 0.05, 0.95)), 0.5, 0.18,
                            (0.95, 0.2, 0.1)))


def _run_two_sides(kw, ticks, brain=None):
    """朝球走：同一模型，球在左 vs 在右各跑一次，横向位移必须不同。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    out = {}
    for tag, x in (("left", 0.15), ("right", 0.85)):
        body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
        body.reset()
        body.data.qpos[2] = 0.24
        body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
        body.mj.mj_forward(body.model, body.data)
        x0, y0 = float(body.data.qpos[0]), float(body.data.qpos[1])
        img = _ball(sp, x)
        for _ in range(ticks):
            eng.clear_external()
            eng.set_input(brain.encoder.encode(_obs(sp, img)))
            eng.step()
            body.step(brain.forces())
        out[tag] = dict(dx=float(body.data.qpos[0]) - x0,
                        dy=float(body.data.qpos[1]) - y0,
                        fallen=bool(body.fallen()))
    dy_side = out["left"]["dy"] - out["right"]["dy"]
    ok = (not out["left"]["fallen"] and not out["right"]["fallen"]
          and abs(out["left"]["dx"]) > 0.02 and abs(dy_side) > 0.01)
    return dict(ok=ok, dx=out["left"]["dx"], dy_side=dy_side,
                fallen=out["left"]["fallen"] or out["right"]["fallen"])


def run_skill(kw, skill, ticks=100, brain=None):
    if skill == "walk_to_ball":
        return _run_two_sides(kw, ticks, brain=brain)

    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()

    if skill in ("getup1", "getup2", "getup"):
        body.data.qpos[2] = 0.10
        body.data.qpos[3:7] = np.array([0.7071, 0.7071, 0.0, 0.0])
        body.data.qpos[7:19] = np.array([0.0, 1.6, -2.6] * 4)
    else:
        body.data.qpos[2] = 0.24
        body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    yaw0 = float(body.state()["yaw"])

    heights, xs, yaws = [], [], []
    for t in range(ticks):
        if skill == "walk_to_ball":
            frac = t / max(1, ticks - 1)
            img = _ball(sp, 0.85 - 0.6 * frac)       # 球从右边逐渐靠近中心
        elif skill in ("turn1", "turn2", "turn_to_ball"):
            img = _ball(sp, 0.85)
        else:
            img = None
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, img)))
        eng.step()
        body.step(brain.forces())
        heights.append(body.state()["height"])
        xs.append(float(body.data.qpos[0]))
        yaws.append(float(body.state()["yaw"]))

    dx = xs[-1] - x0
    dyaw = yaws[-1] - yaw0
    fallen = bool(body.fallen())
    if skill in ("walk0", "walk1", "walk2", "walk3"):   # 难度档：1.5/3/5/10 厘米
        need = {"walk0": 0.015, "walk1": 0.03, "walk2": 0.05, "walk3": 0.10}[skill]
        return dict(ok=(not fallen) and abs(dx) > need, dx=dx, fallen=fallen)
    if skill == "walk_to_ball":
        return dict(ok=(not fallen) and abs(dx) > 0.03, dx=dx, fallen=fallen)
    if skill in ("turn1", "turn2", "turn_to_ball"):
        need = {"turn1": 0.05, "turn2": 0.15, "turn_to_ball": 0.15}[skill]
        return dict(ok=(not fallen) and abs(dyaw) > need, dyaw=dyaw, fallen=fallen)
    if skill in ("getup1", "getup2", "getup"):
        h = float(np.mean(heights[-20:]))
        need = {"getup1": 0.18, "getup2": 0.22, "getup": 0.20}[skill]
        return dict(ok=(not fallen) and h > need, height=h, fallen=fallen)
    return dict(ok=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skill", default="walk", choices=SKILLS)
    p.add_argument("--ticks", type=int, default=100)
    a = p.parse_args()
    print(run_skill({}, a.skill, a.ticks))
