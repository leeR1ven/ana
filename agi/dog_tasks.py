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
          "turn1", "turn2", "getup0", "getup1", "getup2", "follow", "obstacle", "march")


def _brain(kw, preset="small"):
    sp = build_spec(preset, seed=7 + int(kw.get("seed", 0)))
    # 凡是 spec 上有的字段，都按它的默认类型套用（以前只映射了少数几个，
    # 导致 motor_fan_in / wall_inh 这类参数"传了却没生效"）。
    for k, v in kw.items():
        if k == "seed" or not hasattr(sp, k):
            continue
        cur = getattr(sp, k)
        try:
            setattr(sp, k, type(cur)(v))
        except Exception:
            setattr(sp, k, v)
    brain = build(sp, verbose=False)
    _wire_clock(brain, kw)
    _wire_wall(brain, kw)
    return sp, brain


def _wire_wall(brain, kw):
    """挑出"对蓝色障碍特异"的视觉单元，接到墙抑制组。

    位置先验不行（实测无墙 0.22 / 有墙 0.26，分不开），必须按**实际响应**来挑：
    跑一遍"蓝墙 vs 无墙"，取响应差最大的单元，再接抑制组。
    """
    if not kw.get("wall_inh"):
        return
    wid = brain.motor.get("wall_inh")
    if wid is None:
        return
    wid = np.unique(np.asarray(wid, dtype=np.int64).ravel())
    eng = brain.engine
    sp = brain.spec
    vis = np.asarray(brain.layers["vision.L5"].ids, dtype=np.int64)
    px = sp.px

    BLUE = np.array([0.1, 0.2, 0.95], np.float32)

    def act_of(img, ticks=12):
        acc = None
        for _ in range(ticks):
            eng.clear_external()
            eng.set_input(brain.encoder.encode(_obs(sp, img)))
            eng.step()
            a = np.asarray(eng.act)[vis].copy()
            acc = a if acc is None else acc + a
        return acc / ticks

    ball_img = _ball(sp, 0.5)
    wall_img = np.full((px, px, 3), 0.05, np.float32)
    wall_img[px // 2:, px // 3: 2 * px // 3] = BLUE

    a_ball = act_of(ball_img)
    a_wall = act_of(wall_img)
    sel = np.where((a_wall > 0.4) & (a_ball < 0.15))[0]
    if len(sel) < 8:
        sel = np.argsort(a_wall - a_ball)[-24:]
    sel = sel[np.argsort(a_wall[sel])[-64:]]
    pick = vis[sel]

    m = np.isin(eng.dst, wid)
    eng.w[m] = 0.0                     # 旧的按位置输入的边作废
    eng.wp[m] = 0.0
    src = np.repeat(pick.astype(np.int32), len(wid))
    dst = np.tile(wid, len(pick)).astype(np.int32)
    wgt = float(kw.get("wall_inh_in", 1.0))
    eng.add_edges(src, dst, np.full(len(src), wgt, np.float32), group=81)
    # 用绝对阈值：蓝特异单元有墙时约 0.7 亮、无墙时约 0，
    # 所以把门槛设在"三成亮"处，就能把两种情形干净分开。
    n_each = max(1, len(pick))
    eng.set_thr(wid, thr=np.full(len(wid), 0.30 * wgt * n_each, np.float32),
                soft=np.full(len(wid), 0.10 * wgt * n_each, np.float32))
    eng.rebuild_backend()


def _wire_clock(brain, kw):
    """把时间环接进运动区，并让每拍真的推进时钟。

    之前狗任务直接调 engine.step()，绕过了 Loop，时间环恒为 0——
    脑因此只能输出静态姿势（逐拍标准差 0.00000）。
    """
    gain = float(kw.get("clock_motor", 0.0))
    tr = getattr(brain, "time_ring", None)
    if gain <= 0 or tr is None:
        return
    eng = brain.engine
    ring = np.asarray(tr.ids, dtype=np.int64).ravel()
    mpfc = np.asarray(brain.motor["mpfc"], dtype=np.int64).ravel()
    rng = np.random.default_rng(int(kw.get("seed", 0)) + 991)
    src, dst, w = [], [], []
    for m in mpfc:
        for r in rng.choice(ring, 4, replace=False):
            src.append(int(r))
            dst.append(int(m))
            w.append(gain)
    eng.add_edges(np.array(src, np.int32), np.array(dst, np.int32),
                  np.array(w, np.float32), group=78)
    eng.rebuild_backend()
    _orig_set = eng.set_input

    def _set_with_clock(code):
        tr.step(eng)                 # 先推进时钟（写进外部输入）
        return _orig_set(code)       # 再写感觉；顺序反了时钟会被清掉

    eng.set_input = _set_with_clock


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
    if skill == "follow":
        return _run_follow(kw, ticks, brain=brain)
    if skill == "obstacle":
        return _run_obstacle(kw, ticks, brain=brain)
    if skill == "march":
        return _run_march(kw, ticks, brain=brain)

    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()

    if skill in ("getup0", "getup1", "getup2", "getup"):
        # 第 13 条执行器：躯干翻滚（力度由 torso_force 控制）
        body.torso_cmd = float(kw.get("torso_force", 0.0))
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
        if skill in ("walk0", "walk1", "walk2", "walk3"):
            # 偏心的球 = 方向线索；居中画面时模型没有理由动（实测 0.008 vs 0.30）
            img = _ball(sp, 0.85)
        elif skill == "walk_to_ball":
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
    if skill in ("getup0", "getup1", "getup2", "getup"):
        h = float(np.mean(heights[-20:]))
        need = {"getup0": 0.15, "getup1": 0.18, "getup2": 0.22, "getup": 0.20}[skill]
        # 起身本来就是从"躺着"开始，用 fallen 判会永远不过——只看"起来多高"
        return dict(ok=h > need, height=h, fallen=fallen)
    return dict(ok=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skill", default="walk", choices=SKILLS)
    p.add_argument("--ticks", type=int, default=100)
    a = p.parse_args()
    print(run_skill({}, a.skill, a.ticks))

def _ball_moving(sp, x0, x1, frac):
    """球在画面里从 x0 移到 x1（frac ∈ [0,1]）。"""
    return _ball(sp, x0 + (x1 - x0) * frac)


def _run_follow(kw, ticks, brain=None):
    """跟随：球左右扫，两个方向各跑一次，横向位移必须相反（跟着球走）。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    out = {}
    for tag, (a, b) in (("lr", (0.15, 0.85)), ("rl", (0.85, 0.15))):
        body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
        body.reset()
        body.data.qpos[2] = 0.24
        body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
        body.mj.mj_forward(body.model, body.data)
        y0 = float(body.data.qpos[1])
        for t in range(ticks):
            img = _ball_moving(sp, a, b, t / max(1, ticks - 1))
            eng.clear_external()
            eng.set_input(brain.encoder.encode(_obs(sp, img)))
            eng.step()
            body.step(brain.forces())
        out[tag] = float(body.data.qpos[1]) - y0
    diff = out["lr"] - out["rl"]
    return dict(ok=abs(diff) > 0.004, dy_follow=diff, fallen=False)


def _run_obstacle(kw, ticks, brain=None):
    """避障：正前方有黑色障碍时，前进量必须明显小于没有障碍时。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    px = sp.px

    def episode(with_wall):
        body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
        body.reset()
        body.data.qpos[2] = 0.24
        body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
        body.mj.mj_forward(body.model, body.data)
        x0 = float(body.data.qpos[0])
        fs = []
        for _ in range(ticks):
            img = _ball(sp, 0.5)
            if with_wall:
                img = img.copy()
                # 正前方一堵蓝墙：球是红的，蓝色通道能把它和球分开
                img[px // 2:, px // 3: 2 * px // 3] = np.array([0.1, 0.2, 0.95], np.float32)
            eng.clear_external()
            eng.set_input(brain.encoder.encode(_obs(sp, img)))
            eng.step()
            f = brain.forces()
            fs.append(float(np.mean(f)))
            body.step(f)
        return float(body.data.qpos[0]) - x0, float(np.mean(fs))

    d0, f0 = episode(False)
    d1, f1 = episode(True)
    # 判据改成"看见障碍就收力"——位移主要来自滑动，用它判测不出东西
    ok = f1 < f0 * 0.75
    return dict(ok=ok, dx_free=d0, dx_wall=d1, force_free=f0, force_wall=f1, fallen=False)

def _run_march(kw, ticks, brain=None):
    """行进（画面交替）：左右两张图逐拍交替，看能不能持续前进。

    这个和 walk0 的区别：walk0 给静止画面，模型几乎不动；
    walk_to_ball/follow 给变化的画面，模型能走 0.29 米。
    所以"会不会走"要用**变化**的输入来测。
    """
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    imgs = (_ball(sp, 0.2), _ball(sp, 0.8))
    for t in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, imgs[t % 2])))
        eng.step()
        body.step(brain.forces())
    dx = float(body.data.qpos[0]) - x0
    return dict(ok=(not body.fallen()) and abs(dx) > 0.03, dx=dx, fallen=bool(body.fallen()))
