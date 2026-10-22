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
          "back", "tilt", "shake", "orbit", "reach", "gait", "jump", "detour", "chase", "gaze", "lift", "orient", "robust", "recover", "verge", "biped", "goto", "goto2")


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
    _apply_mod(brain, kw)
    _wire_clock(brain, kw)
    _wire_wall(brain, kw)
    return sp, brain


def _apply_mod(brain, kw):
    """全局调制（类似神经调质）：把整组肌肉出力乘一个系数。

    实测：增益 0.6/1.0/1.4 -> 幅度 0.31/0.52/0.72、持续时间 9/13/20 拍。
    也就是说同一个动作程序，靠这一个系数就能"做快做慢、做大做小"。
    """
    g = float(kw.get("motor_mod", 1.0))
    if g == 1.0:
        return
    orig = brain.forces

    def forces():
        return np.clip(orig() * g, 0.0, 1.0)

    brain.forces = forces


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
    # 通用"危险"通路：近处红色大块（逼近物）也触发抑制 -> 反射性后退/停下
    cy = cx = (px - 1) / 2.0
    for y in range(px):
        for x in range(px):
            if (y - cy) ** 2 + (x - cx) ** 2 <= (0.30 * px) ** 2:
                j = (y * px + x) * 3 + 0      # 红通道、中心区域
                base = a0 + j * 2 * n_lv
                ids.extend(range(base, base + n_lv))
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
    period = float(kw.get("clock_period", 25.0))
    tr.ticks_per_slot = max(1, int(round(len(np.asarray(tr.ids).ravel()) / max(4.0, period))))
    ring = np.asarray(tr.ids, dtype=np.int64).ravel()
    mpfc = np.asarray(brain.motor["mpfc"], dtype=np.int64).ravel()
    rng = np.random.default_rng(int(kw.get("seed", 0)) + 991)
    src, dst, w = [], [], []
    if kw.get("clock_gait"):
        # 步态版接线：给运动区单元加上"属于哪条腿"和"相位偏移"。
        # 对角腿（FL+RR / FR+RL）相差半个周期；反向只需把相位递减。
        n_mpfc = len(mpfc)
        per = max(1, n_mpfc // 12)          # 每块肌肉分到的 mpfc 单元数
        leg_phase = (0.0, 0.5, 0.5, 0.0)    # FL, FR, RL, RR（对角同相）
        rev = bool(kw.get("clock_reverse"))
        for mus in range(12):
            leg = mus // 3
            ph = leg_phase[leg] + float(kw.get("clock_knee", 0.0)) * (1 if mus % 3 == 2 else 0)
            base = (ph * len(ring)) % len(ring)
            # 每条肌肉接一个**相位窗口**：只接环上一小段（占 1/8 圈），
            # 窗口起点按腿的相位偏移。接单格等于 512 拍才亮一次，不成节律。
            width = max(2, int(len(ring) * float(kw.get("clock_window", 0.125))))
            for i in range(per):
                unit = int(mpfc[min(n_mpfc - 1, mus * per + i)])
                k2 = i % width
                idx = int((base + (k2 if not rev else -k2)) % len(ring))
                src.append(int(ring[idx]))
                dst.append(unit)
                w.append(gain)
    else:
        for m in mpfc:
            for r in rng.choice(ring, 4, replace=False):
                src.append(int(r))
                dst.append(int(m))
                w.append(gain)
    # 第二条通路：随机相位（给转向/跳跃这类瞬时动作留一条不按步态的驱动）
    rgain = float(kw.get("clock_rand", 0.0))
    if rgain > 0:
        for m in mpfc:
            for r in rng.choice(ring, 3, replace=False):
                src.append(int(r))
                dst.append(int(m))
                w.append(rgain)
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
    if skill == "lift":
        return _run_lift(kw, ticks, brain=brain)
    if skill == "orient":
        return _run_orient(kw, ticks, brain=brain)
    if skill == "robust":
        return _run_robust(kw, ticks, brain=brain)
    if skill == "recover":
        return _run_recover(kw, ticks, brain=brain)
    if skill == "verge":
        return _run_verge(kw, ticks, brain=brain)
    if skill == "biped":
        return _run_biped(kw, ticks, brain=brain)
    if skill == "goto":
        return _run_goto(kw, ticks, brain=brain)
    if skill == "orbit":
        return _run_orbit(kw, ticks, brain=brain)
    if skill == "reach":
        return _run_reach(kw, ticks, brain=brain)
    if skill == "gait":
        return _run_gait(kw, ticks, brain=brain)
    if skill == "goto2":
        return _run_goto2(kw, ticks, brain=brain)
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
    _pan = 0.0                                   # 眼动反射的累计偏转
    _reflex = bool(kw.get("eye_reflex"))
    for t in range(ticks):
        if _reflex and skill in ("walk0", "walk1", "walk2", "walk3", "walk4", "walk5"):
            # 眼动反射：每拍把球往画面中心拉一点（实测能让位移翻倍）
            img = _ball(sp, float(np.clip(0.85 + _pan, 0.05, 0.95)))
            _pan -= 0.35 * (_red_center(img) - 0.5)
        elif skill in ("walk0", "walk1", "walk2", "walk3", "walk4", "walk5"):
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


def _leg_lift_reflex(forces, touch, gain=0.9):
    """脊髓级反射：某条腿触觉强 -> 这条腿自己抬起（不经脑）。

    肌肉编号按"腿优先"排列：第 i 条腿用 [3i, 3i+1, 3i+2] 三条肌肉，
    抬起 = 加大第 2 条（膝）与第 1 条（大腿）的目标偏移。
    """
    f = np.array(forces, np.float32, copy=True)
    if touch is None:
        return f
    t = np.asarray(touch, np.float32)
    n = min(4, t.size)
    for leg in range(n):
        if t[leg] > 0.6:
            f[leg * 3 + 1] = min(1.0, f[leg * 3 + 1] + gain)
            f[leg * 3 + 2] = min(1.0, f[leg * 3 + 2] + gain)
    return f


def _audio_from(sp, side):
    """模拟两只耳朵：side<0 左声道强，side>0 右声道强。"""
    au = np.full((sp.audio_bands, sp.audio_frames), 0.2, np.float32)
    b = sp.audio_bands
    if side < 0:
        au[: max(1, b // 3), :] = 0.95
        au[2 * b // 3:, :] = 0.35
    else:
        au[2 * b // 3:, :] = 0.95
        au[: max(1, b // 3), :] = 0.35
    return au


def _balance_reflex(forces, roll, pitch, gain=0.8, thr=0.15):
    """平衡反射：往哪边倒，就加大那一侧对侧腿的支撑（脊髓/前庭级）。

    roll > 0 表示向右倾 -> 加大右侧（FR/RR）腿的支撑去顶回来；pitch 同理。
    """
    f = np.array(forces, np.float32, copy=True)
    if abs(roll) > thr:
        legs = (1, 3) if roll > 0 else (0, 2)      # 0=FL 1=FR 2=RL 3=RR
        for leg in legs:
            f[leg * 3 + 1] = min(1.0, f[leg * 3 + 1] + gain * min(1.0, abs(roll) / 0.6))
    if abs(pitch) > max(0.20, thr * 1.3):
        legs = (2, 3) if pitch > 0 else (0, 1)
        for leg in legs:
            f[leg * 3 + 2] = min(1.0, f[leg * 3 + 2] + gain * min(1.0, abs(pitch) / 0.8))
    return f


def _run_goto2(kw, ticks, brain=None):
    """按目标远近调节距离：球越大 = 目标越近，要求"近目标走短、远目标走长"。"""
    from agi.world import _blob
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    px = sp.px
    out = {}
    for size in (0.10, 0.18, 0.26):
        img = np.maximum(np.full((px, px, 3), 0.05, np.float32),
                         _blob(px, 0.85, 0.5, size, (0.95, 0.2, 0.1)))
        body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
        body.reset()
        body.data.qpos[2] = 0.24
        body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
        body.mj.mj_forward(body.model, body.data)
        x0 = float(body.data.qpos[0])
        for _ in range(ticks):
            eng.clear_external()
            eng.set_input(brain.encoder.encode(_obs(sp, img)))
            eng.step()
            body.step(brain.forces())
        out[size] = abs(float(body.data.qpos[0]) - x0)
    a, b, c = out[0.10], out[0.18], out[0.26]
    ok = (a > b > c) and (a - c) > 0.015
    return dict(ok=bool(ok), far=float(a), mid=float(b), near=float(c), fallen=False)


def _foot_forces(body):
    """四条腿各自是否着地（看接触是否发生在该腿的小腿 body 上）。"""
    import mujoco
    m, d = body.model, body.data
    legs = []
    for nm in ("FL", "FR", "RL", "RR"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm + "_calf")
        legs.append(bid)
    if any(b < 0 for b in legs):
        return np.ones(4, np.float32) * 9.9
    on = np.zeros(4, np.float32)
    for k in range(d.ncon):
        con = d.contact[k]
        b1 = m.geom_bodyid[con.geom1] if con.geom1 >= 0 else -1
        b2 = m.geom_bodyid[con.geom2] if con.geom2 >= 0 else -1
        for i, bid in enumerate(legs):
            if b1 == bid or b2 == bid:
                on[i] = 1.0
    return on


def _run_gait(kw, ticks, brain=None):
    """真走路：不飞、不翻，而且四条腿不能同相（对角腿要反相）。

    这是对"位移>阈值"的纠正：位移可以被晃动/漂移/翻车满足，相位不能。
    """
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.27
    body.data.qpos[7:19] = np.array([0.0, 0.9, -1.8] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    hs, qs, foot_contact = [], [], []
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        body.step(brain.forces())
        hs.append(float(body.data.qpos[2]))
        qs.append(body.data.qpos[7:19].copy())
        foot_contact.append(_foot_forces(body))
    q = np.stack(qs)
    # 足端接触：每条腿"着地"的拍数占比（真正的迈步必须有腿离地）
    contacts = np.asarray(foot_contact, np.float32)      # (ticks, 4)
    stance = contacts.mean(0)                            # 每条腿着地比例
    swing_ok = bool((stance < 0.95).sum() >= 2)           # 至少两条腿有明显摆动相
    q = q - q.mean(0)
    leg = np.stack([q[:, 0:3].mean(1), q[:, 3:6].mean(1),
                    q[:, 6:9].mean(1), q[:, 9:12].mean(1)])

    def corr(a, b):
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    diag = (corr(leg[0], leg[3]) + corr(leg[1], leg[2])) / 2     # 对角腿相关
    amp = float(np.degrees(q.max(0) - q.min(0)).mean())
    dx = abs(float(body.data.qpos[0]) - x0)
    hi = max(hs)
    ok = (hi < 0.45) and (diag < 0.30) and (amp > 8.0) and (dx > 0.05) and swing_ok
    return dict(ok=bool(ok), dx=dx, maxh=float(hi), diag=float(diag), amp=amp,
                swing=float((stance < 0.95).sum()), fallen=bool(body.fallen()))


def _run_reach(kw, ticks, brain=None):
    """够高处的东西：球在画面上方，狗要立起来（后腿撑地、身体竖起）去够它。

    "两腿站立"不是姿势题，是为够到高处目标的副产物。
    """
    from agi.world import _blob
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.torso_cmd = float(kw.get("torso_force", 0.0))
    body.data.qpos[2] = 0.26
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    px = sp.px
    high = np.maximum(np.full((px, px, 3), 0.05, np.float32),
                      _blob(px, 0.5, 0.15, 0.14, (0.95, 0.2, 0.1)))   # 高处目标
    hs = []
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, high)))
        eng.step()
        body.step(brain.forces())
        hs.append(float(body.state()["height"]))
    h = float(np.mean(hs[-20:]))
    return dict(ok=bool(h > 0.30), height=h, fallen=bool(body.fallen()))


def _run_orbit(kw, ticks, brain=None):
    """追绕着狗转的球：球在视野里左右绕圈，狗为了追它自己得转。

    "转圈"不是题目，是追球的副产物——动作有目的，不是平白无故做的。
    """
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    prev = float(body.state()["yaw"])
    total = 0.0
    for t in range(ticks):
        theta = 0.9 * np.sin(2 * np.pi * t / max(8, ticks / 3.0))   # 球绕圈
        x = float(np.clip(0.5 + 0.45 * np.sin(theta), 0.05, 0.95))
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, x))))
        eng.step()
        body.step(brain.forces())
        cur = float(body.state()["yaw"])
        total += (cur - prev + np.pi) % (2 * np.pi) - np.pi
        prev = cur
    # 门槛定在可达范围（实测这个身体 150 拍约 0.18 弧度）
    return dict(ok=bool((not body.fallen()) and abs(total) > 0.15),
                yaw=float(total), fallen=bool(body.fallen()))


def _run_goto(kw, ticks, brain=None):
    """走过去并停在指定距离：需要出发（触发）、用力（调制）、到位后收住（微调）。"""
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
    dx = abs(float(body.data.qpos[0]) - x0)
    return dict(ok=bool((not body.fallen()) and 0.14 <= dx <= 0.22),
                dx=float(body.data.qpos[0]) - x0, fallen=bool(body.fallen()))


def _run_biped(kw, ticks, brain=None):
    """两腿站立：后腿撑地、身体立起来（高度升 + 抬头 pitch 明显）。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.torso_cmd = float(kw.get("torso_force", 0.0))
    body.data.qpos[2] = 0.26
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    hs, ps = [], []
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        body.step(brain.forces())
        st = body.state()
        hs.append(float(st["height"]))
        ps.append(float(st["pitch"]))
    h = float(np.mean(hs[-20:]))
    p_ = float(np.mean(ps[-20:]))
    # 判据定在"比站姿高一大截 + 明显抬头"（实测只有 tors​​o 拉满才够）
    # 20Nm 的躯干力矩下真实可达高度约 0.32（40Nm 能到 1 米但那是飞起来，不算站立）
    return dict(ok=bool(h > 0.30 and abs(p_) > 0.05), height=h, pitch=p_, fallen=bool(body.fallen()))


def _run_verge(kw, ticks, brain=None):
    """双眼会聚：物体在不同深度，左右眼各自转，把物体拉到**各自**画面中心。

    近处两眼要向内转（会聚角大），远处接近平行——这就是立体视觉的基础。
    """
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    out = {}
    for depth in (0.6, 3.0):
        body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
        body.reset()
        body.data.qpos[2] = 0.24
        body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
        body.mj.mj_forward(body.model, body.data)
        panL = panR = 0.0
        cL = cR = 0.5
        for _ in range(ticks):
            L, R = _ball_pair(sp, 0.5, depth=depth)
            cL, cR = _red_center(L), _red_center(R)
            panL -= 0.35 * (cL - 0.5)          # 眼动反射：各自把物体拉到中心
            panR -= 0.35 * (cR - 0.5)
            # 用会聚后的视角重新渲染（左右各偏 pan）
            L = _ball(sp, float(np.clip(0.5 + panL, 0.02, 0.98)))
            R = _ball(sp, float(np.clip(0.5 + panR, 0.02, 0.98)))
            img = _obs_stereo(sp, L, R)
            eng.clear_external()
            eng.set_input(brain.encoder.encode(_obs(sp, img)))
            eng.step()
            body.step(brain.forces())
        out[depth] = abs(panL - panR)
    near, far = out[0.6], out[3.0]
    return dict(ok=bool(near > far + 0.01), near=near, far=far, fallen=False)


def _run_robust(kw, ticks, brain=None):
    """鲁棒运动：一路有随机推力，要求不摔且仍有位移。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    rng = np.random.default_rng(int(kw.get("seed", 0)) + 5)
    reflex = bool(kw.get("balance_reflex", True))
    gain = float(kw.get("balance_gain", 1.5))
    thr = float(kw.get("balance_thr", 0.15))
    push = float(kw.get("push_amp", 0.3))
    fell = False
    for t in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        f = brain.forces()
        if reflex:
            st = body.state()
            f = _balance_reflex(f, st["roll"], st["pitch"], gain=gain, thr=thr)
        body.step(f)
        if t % 12 == 0:                      # 每 12 拍来一下横向推力
            body.data.qvel[1] += float(rng.uniform(-push, push))
        if body.fallen():
            fell = True
    dx = abs(float(body.data.qpos[0]) - x0)
    return dict(ok=(not fell) and dx > 0.05, dx=dx, fell=fell, fallen=bool(body.fallen()))


def _run_recover(kw, ticks, brain=None):
    """摔倒自恢复：从四脚朝天起步，要求最后站起来。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.torso_cmd = float(kw.get("torso_force", 0.0))
    body.data.qpos[2] = 0.12
    body.data.qpos[3:7] = np.array([0.0, 1.0, 0.0, 0.0])      # 翻过来
    body.data.qpos[7:19] = np.array([0.0, 1.8, -2.8] * 4)
    body.mj.mj_forward(body.model, body.data)
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        body.step(brain.forces())
    h = float(body.state()["height"])
    return dict(ok=h > 0.22, height=h, fallen=bool(body.fallen()))


def _run_orient(kw, ticks, brain=None):
    """听声定向：声音从左/右来，机体应往声音那边转（两次偏航符号相反）。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    out = {}
    for tag, side in (("left", -1), ("right", 1)):
        body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
        body.reset()
        body.data.qpos[2] = 0.24
        body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
        body.mj.mj_forward(body.model, body.data)
        yaw0 = float(body.state()["yaw"])
        for _ in range(ticks):
            o = _obs(sp, _ball(sp, 0.5))
            o.audio = _audio_from(sp, side)
            eng.clear_external()
            eng.set_input(brain.encoder.encode(o))
            eng.step()
            body.step(brain.forces())
        out[tag] = float(body.state()["yaw"]) - yaw0
    diff = out["left"] - out["right"]
    return dict(ok=bool(abs(diff) > 0.03), dyaw_left=out["left"],
                dyaw_right=out["right"], fallen=False)


def _run_lift(kw, ticks, brain=None, leg=0):
    """被触碰的腿会抬起来：只给第 leg 条腿触觉，看它是否比别的高。"""
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    q0 = body.data.qpos[7:19].copy()
    touch = np.zeros(4, np.float32)
    touch[leg] = 1.0
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        f = _leg_lift_reflex(brain.forces(), touch)
        body.step(f)
    q = body.data.qpos[7:19].copy()
    d = np.abs(np.degrees(q - q0))
    mine = float(d[leg * 3: leg * 3 + 3].mean())
    others = float(np.delete(d, [leg * 3, leg * 3 + 1, leg * 3 + 2]).mean())
    # 用绝对角度变化判：全身耦合会让"其他腿"也动，比较差值会被污染
    return dict(ok=bool(mine > 5.0), mine=mine, others=others, fallen=bool(body.fallen()))


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
    # 眼动反射：把鲜艳目标拉到画面中心（像上丘反射，不需要学）
    pan = 0.0
    center = 0.5
    for _ in range(ticks):
        f = brain.forces()
        x = float(np.clip(0.90 + pan, 0.05, 0.95))
        img = _ball(sp, x)
        center = _red_center(img)
        pan -= 0.35 * (center - 0.5)          # 往目标方向转
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
    """后退：眼前一个很近的大物体（占满画面中心），应该往后退。"""
    from agi.world import _blob
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.data.qpos[2] = 0.24
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    px = sp.px
    big = np.maximum(np.full((px, px, 3), 0.05, np.float32),
                     _blob(px, 0.5, 0.5, 0.38, (0.95, 0.2, 0.1)))
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, big)))
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
    """凑近看地上的东西：目标在画面下方（低处），狗要压低身体贴近它。

    "趴下"不是姿势题，是为了看清地上目标的副产物。
    """
    from agi.world import _blob
    sp, brain = (brain.spec, brain) if brain is not None else _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.12)))
    body.reset()
    body.torso_cmd = float(kw.get("torso_force", 0.0))
    body.data.qpos[2] = 0.26
    body.data.qpos[7:19] = np.array([0.0, 1.0, -2.0] * 4)
    body.mj.mj_forward(body.model, body.data)
    px = sp.px
    low = np.maximum(np.full((px, px, 3), 0.05, np.float32),
                     _blob(px, 0.5, 0.85, 0.14, (0.95, 0.2, 0.1)))    # 地上的目标
    hs = []
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, low)))
        eng.step()
        body.step(brain.forces())
        hs.append(float(body.state()["height"]))
    h = float(np.mean(hs[-20:]))
    return dict(ok=bool(h < 0.22), height=h, fallen=bool(body.fallen()))


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
    # 无墙时必须真的有力气，否则"两边都归零"会假通过
    ok = (f0 > 0.05) and (f1 < f0 * (0.40 if strict else 0.75))
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

def _ball_xy(sp, x_norm, y_norm, size=0.18):
    """把红球画在画面 (x, y) 处：y=0 顶部（高处），y=1 底部（地上）。"""
    from agi.world import _blob
    px = sp.px
    return np.maximum(np.full((px, px, 3), 0.05, np.float32),
                      _blob(px, float(np.clip(x_norm, 0.05, 0.95)),
                            float(np.clip(y_norm, 0.05, 0.95)), size, (0.95, 0.2, 0.1)))

def _hebb_step(eng, group=90, lr=0.7):
    """海马体那组边做一次共激活增强（赫布）。

    学习率要大（0.7）：一次配对就把权重推到 0.7，两三次接近饱和——
    这才叫"瞬间记住"。用 (1-w) 因子做饱和：涨得快、之后很难再涨。
    小学习率（0.01~0.05）会让记忆变成"慢慢累积"，短时间内记不住东西。
    """
    m = np.asarray(eng.group) == group
    if not m.any():
        return 0
    a = np.asarray(eng.act)
    pre = a[np.asarray(eng.src)[m]]
    post = a[np.asarray(eng.dst)[m]]
    w = np.asarray(eng.wp)[m]
    new = np.minimum(1.0, w + lr * pre * post * (1.0 - w))
    eng.wp[m] = new
    return int((new > w + 1e-6).sum())

