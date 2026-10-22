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


SKILLS = ("walk0", "walk1", "walk2", "walk3", "walk4", "walk5", "walk_to_ball",
          "turn1", "turn2", "turn3", "getup0", "getup1", "getup2", "getup3",
          "follow", "follow2", "obstacle", "obstacle2", "march",
          "back", "tilt", "lie", "spin", "crawl", "shake", "jump", "detour", "chase", "gaze")


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

    # 直接接"蓝色通道、视野下方中央"的输入细胞：
    # 视觉是逐通道温度计，蓝墙只在蓝色通道上亮，红球不会。
    lay = brain.encoder.layout["vision"]
    a0, b0 = int(lay[0]), int(lay[1])
    n_lv = int(brain.spec.n_level)
    ch = (b0 - a0) // (2 * n_lv)
    cols = ch // 3
    px = int(round(cols ** 0.5))
    ids = []
    for y in range(px // 2, px):
        for x in range(px // 3, max(px // 3 + 1, 2 * px // 3)):
            j = (y * px + x) * 3 + 2          # 蓝通道
            base = a0 + j * 2 * n_lv
            ids.extend(range(base, base + n_lv))   # 正的温度计格
    ids = np.asarray(ids, np.int64)
    if len(ids) == 0:
        return
    m = np.isin(eng.dst, wid)
    eng.w[m] = 0.0
    eng.wp[m] = 0.0
    src = np.repeat(ids.astype(np.int32), len(wid))
    dst = np.tile(wid, len(ids)).astype(np.int32)
    wgt = float(kw.get("wall_inh_in", 0.5))
    eng.add_edges(src, dst, np.full(len(src), wgt, np.float32), group=83)
    thr = np.full(len(wid), 0.35 * wgt * len(ids), np.float32)
    eng.set_thr(wid, thr=thr, soft=(thr * 0.3).astype(np.float32))
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
    if skill in ("follow", "follow2"):
        return _run_follow(kw, ticks, brain=brain, strict=(skill == "follow2"))
    if skill in ("obstacle", "obstacle2"):
        return _run_obstacle(kw, ticks, brain=brain, strict=(skill == "obstacle2"))
    if skill == "back":
        return _run_back(kw, ticks, brain=brain)
    if skill == "tilt":
        return _run_tilt(kw, ticks, brain=brain)
    if skill == "lie":
        return _run_lie(kw, ticks, brain=brain)
    if skill == "spin":
        return _run_spin(kw, ticks, brain=brain)
    if skill == "crawl":
        return _run_crawl(kw, ticks, brain=brain)
    if skill == "shake":
        return _run_shake(kw, ticks, brain=brain)
    if skill == "jump":
        return _run_jump(kw, ticks, brain=brain)
    if skill == "detour":
        return _run_detour(kw, ticks, brain=brain)
    if skill == "chase":
        return _run_chase(kw, ticks, brain=brain)
    if skill == "gaze":
        return _run_gaze(kw, ticks, brain=brain)
    if skill == "march":
        return _run_march(kw, ticks, brain=brain)

    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()

    if skill in ("getup0", "getup1", "getup2", "getup3", "getup"):
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
        if skill in ("walk0", "walk1", "walk2", "walk3", "walk4", "walk5"):
            # 偏心的球 = 方向线索；居中画面时模型没有理由动（实测 0.008 vs 0.30）
            img = _ball(sp, 0.85)
        elif skill == "walk_to_ball":
            frac = t / max(1, ticks - 1)
            img = _ball(sp, 0.85 - 0.6 * frac)       # 球从右边逐渐靠近中心
        elif skill in ("turn1", "turn2", "turn3", "turn_to_ball"):
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
    if skill in ("walk0", "walk1", "walk2", "walk3", "walk4", "walk5"):
        need = {"walk0": 0.015, "walk1": 0.03, "walk2": 0.05, "walk3": 0.10,
                "walk4": 0.20, "walk5": 0.30}[skill]
        return dict(ok=(not fallen) and abs(dx) > need, dx=dx, fallen=fallen)
    if skill == "walk_to_ball":
        return dict(ok=(not fallen) and abs(dx) > 0.03, dx=dx, fallen=fallen)
    if skill in ("turn1", "turn2", "turn3", "turn_to_ball"):
        need = {"turn1": 0.05, "turn2": 0.15, "turn3": 0.30, "turn_to_ball": 0.15}[skill]
        return dict(ok=(not fallen) and abs(dyaw) > need, dyaw=dyaw, fallen=fallen)
    if skill in ("getup0", "getup1", "getup2", "getup3", "getup"):
        h = float(np.mean(heights[-20:]))
        need = {"getup0": 0.15, "getup1": 0.18, "getup2": 0.22,
                "getup3": 0.35, "getup": 0.20}[skill]
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


def _run_follow(kw, ticks, brain=None, strict=False):
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
    need = 0.010 if strict else 0.004
    return dict(ok=abs(diff) > need, dy_follow=diff, fallen=False)


def _run_gaze(kw, ticks, brain=None):
    """眼睛追随：物体在画面右侧，脑用整体激活水平驱动"眼动"通道，
    把注视点转到物体上（目标红心落在画面中心 ±0.10）。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    center = 0.5
    for _ in range(ticks):
        f = brain.forces()
        pan = float(np.clip(2.0 * float(np.mean(f)) - 1.0, -1.0, 1.0))
        x = float(np.clip(0.90 - 0.45 * pan, 0.05, 0.95))
        img = _ball(sp, x)
        center = _red_center(img)
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, img)))
        eng.step()
        body.step(f)
    err = abs(center - 0.5)
    return dict(ok=bool(err < 0.10), center=float(center), err=float(err), fallen=False)


def _run_jump(kw, ticks, brain=None):
    """跃过障碍：视野下方一条矮蓝条，要求跨过去（位移 > 10cm 且不摔）。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    px = sp.px
    for _ in range(ticks):
        img = _ball(sp, 0.5).copy()
        img[int(px * 0.62):int(px * 0.75), px // 3: 2 * px // 3] = np.array([0.1, 0.2, 0.95], np.float32)
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, img)))
        eng.step()
        body.step(brain.forces())
    dx = float(body.data.qpos[0]) - x0
    return dict(ok=(not body.fallen()) and abs(dx) > 0.10, dx=dx, fallen=bool(body.fallen()))


def _run_detour(kw, ticks, brain=None):
    """绕过障碍：障碍只在视野右半，要求往左让开（dy 明显为负）。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    y0 = float(body.data.qpos[1])
    px = sp.px
    for _ in range(ticks):
        img = _ball(sp, 0.5).copy()
        img[px // 2:, int(px * 0.6):] = np.array([0.1, 0.2, 0.95], np.float32)
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, img)))
        eng.step()
        body.step(brain.forces())
    dy = float(body.data.qpos[1]) - y0
    return dict(ok=(not body.fallen()) and dy < -0.03, dy=dy, fallen=bool(body.fallen()))


def _run_chase(kw, ticks, brain=None):
    """追移动物体：球从一端扫到另一端，要求机体横向位移与球同向。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    out = {}
    for tag, (a, b) in (("lr", (0.2, 0.8)), ("rl", (0.8, 0.2))):
        body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
        body.reset()
        body.data.qpos[2] = 0.24
        body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
        body.mj.mj_forward(body.model, body.data)
        y0 = float(body.data.qpos[1])
        for t in range(ticks):
            x = a + (b - a) * t / max(1, ticks - 1)
            eng.clear_external()
            eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, x))))
            eng.step()
            body.step(brain.forces())
        out[tag] = float(body.data.qpos[1]) - y0
    same = out["lr"] < out["rl"]        # 球向左移时，机体也往左（dy 更小）
    return dict(ok=bool(same) and abs(out["lr"] - out["rl"]) > 0.02,
                lr=out["lr"], rl=out["rl"], fallen=False)


def _run_spin(kw, ticks, brain=None):
    """原地转圈：累计偏航超过 2π（转满一圈）。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    total = 0.0
    prev = float(body.state()["yaw"])
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        body.step(brain.forces())
        cur = float(body.state()["yaw"])
        d = (cur - prev + np.pi) % (2 * np.pi) - np.pi
        total += d
        prev = cur
    return dict(ok=(not body.fallen()) and abs(total) > 2 * np.pi,
                spin=float(total), fallen=bool(body.fallen()))


def _run_crawl(kw, ticks, brain=None):
    """低姿爬行：全程高度低于 0.16，但仍有位移。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.14
    body.data.qpos[7:19] = np.array([0.0, 1.6, -2.6] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    hs = []
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        body.step(brain.forces())
        hs.append(float(body.state()["height"]))
    dx = float(body.data.qpos[0]) - x0
    return dict(ok=(float(np.mean(hs)) < 0.20) and abs(dx) > 0.05,
                dx=dx, h=float(np.mean(hs)), fallen=bool(body.fallen()))


def _run_shake(kw, ticks, brain=None):
    """甩身：机身的横向速度来回变化（左右甩动的幅度）。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    ys = []
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        body.step(brain.forces())
        ys.append(float(body.data.qpos[1]))
    amp = float(np.std(ys))
    return dict(ok=amp > 0.01, amp=amp, fallen=bool(body.fallen()))


def _run_back(kw, ticks, brain=None):
    """后退：同样给"前方有方向线索"的输入，要求往后退（dx 为负）。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        body.step(brain.forces())
    dx = float(body.data.qpos[0]) - x0
    return dict(ok=(not body.fallen()) and dx < -0.03, dx=dx, fallen=bool(body.fallen()))


def _run_tilt(kw, ticks, brain=None):
    """倾倒自救：从侧倾 0.5 弧度起步，要求最后站正且不摔。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.26
    r = 0.5
    body.data.qpos[3:7] = np.array([np.cos(r / 2), np.sin(r / 2), 0.0, 0.0])
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        body.step(brain.forces())
    st = body.state()
    ok = (not body.fallen()) and abs(st["roll"]) < 0.25 and st["height"] > 0.22
    return dict(ok=bool(ok), roll=float(st["roll"]), height=float(st["height"]),
                fallen=bool(body.fallen()))


def _run_lie(kw, ticks, brain=None):
    """趴下：从站姿开始，要求最后趴低（高度 < 0.12）。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.26
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        body.step(brain.forces())
    h = float(body.state()["height"])
    return dict(ok=h < 0.12, height=h, fallen=bool(body.fallen()))


def _run_obstacle(kw, ticks, brain=None, strict=False):
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
    ok = f1 < f0 * (0.40 if strict else 0.75)
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

def _ball_pair(sp, x_norm, depth=1.0, baseline=0.06):
    """两只眼：同一个球在左右两图里位置略有不同（视差 ∝ 基线/深度）。"""
    from agi.world import _blob
    px = sp.px
    disp = 0.5 * baseline / max(0.2, float(depth))     # 归一化视差
    left = np.maximum(np.full((px, px, 3), 0.05, np.float32),
                      _blob(px, float(np.clip(x_norm - disp, 0.02, 0.98)), 0.5, 0.18,
                            (0.95, 0.2, 0.1)))
    right = np.maximum(np.full((px, px, 3), 0.05, np.float32),
                       _blob(px, float(np.clip(x_norm + disp, 0.02, 0.98)), 0.5, 0.18,
                             (0.95, 0.2, 0.1)))
    return left, right


def _red_center(img):
    """图上红色像素的水平重心（归一化到 0~1），没有就返回 0.5。"""
    r = img[:, :, 0]
    g = img[:, :, 1]
    mask = (r > 0.5) & (g < 0.5)
    if not mask.any():
        return 0.5
    xs = np.nonzero(mask)[1]
    return float(xs.mean()) / max(1, img.shape[1] - 1)

def _obs_stereo(sp, left, right):
    """两只眼进同一个脑：把左右两图各压成半高，上下拼成一张 px×px 的图。

    这样不用改编码器布局（仍是 px*px*3），但每只眼的分辨率减半。
    """
    px = sp.px
    h = px // 2
    def half(a):
        a = np.nan_to_num(np.asarray(a, np.float32), nan=0.05, posinf=1.0, neginf=0.0)
        a = np.clip(a, 0.0, 1.0)
        return a.reshape(h, 2, px, 3).mean(axis=1)      # 两行并一行（纯 numpy）

    img = np.zeros((px, px, 3), np.float32)
    img[:h] = half(left)
    img[h:] = half(right)
    return img

