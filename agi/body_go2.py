# -*- coding: utf-8 -*-
"""宇树 Go2 的身体接口（MuJoCo）。

只做两件事：
  · `observe()`：把机体状态变成架构要的五种感觉（本体、平衡、触地；视觉/听觉先给零）；
  · `step(forces)`：先用 PD 把站姿撑住，再把大脑给的 12 个"力度档"当成**附加偏置**
    叠到目标关节角上（v0 力度还很小，所以不会把狗推倒）。

关键约定（老库踩过的坑）：
  · 物理 500 Hz，大脑 50 Hz：一拍里跑 10 个物理子步；
  · 力矩要限幅、关节要在限位内；
  · 站立靠 PD（静力学问题），动态步态以后要靠"状态→动作"的连接，不是开环轨迹。

用法（需要装了 mujoco 的 Python）：
    from agi.body_go2 import Go2Body
    body = Go2Body(); body.reset(); body.step();  print(body.state())
"""
from __future__ import annotations

import os

import numpy as np

from .encoding import Observation

DEFAULT_XML = os.environ.get("ANA_GO2_XML") or r"C:\mujoco_models\unitree_go2\scene.xml"


class Go2Body:
    """12 关节力矩电机 + PD 站姿保持。"""

    def __init__(self, xml_path: str = DEFAULT_XML, kp: float = 60.0,
                 kd: float = 2.0, physics_hz: int = 500, brain_hz: int = 50,
                 posture_offset: float = 0.0):
        import mujoco  # 延迟导入：没有 mujoco 时其它模块照常可用
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.n_muscle = 12
        self.kp = float(kp)
        self.kd = float(kd)
        self.substeps = max(1, int(round(physics_hz / brain_hz)))
        self.posture_offset = float(posture_offset)
        self.model.opt.timestep = 1.0 / physics_hz
        # 站姿：优先用模型自带的 home 关键帧
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.q_stand = self.data.qpos[7:19].copy()
        else:
            self.q_stand = np.zeros(12)
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self.jnt_lo = self.model.jnt_range[1:13, 0].copy()
        self.jnt_hi = self.model.jnt_range[1:13, 1].copy()
        self.forces = np.zeros(12, np.float32)

    # ------------------------------------------------------------ 控制
    def reset(self) -> None:
        self.mj.mj_resetDataKeyframe(self.model, self.data, 0)
        for _ in range(20):
            self._pd_step()
            self.mj.mj_step(self.model, self.data)

    def _pd_step(self) -> None:
        if getattr(self, 'torso', 0.0):
            # 第 13 条执行器：躯干翻滚力矩（狗本来就有躯干肌）
            self.data.xfrc_applied[1, 3] = self.torso * 40.0
        q = self.data.qpos[7:19]
        qd = self.data.qvel[6:18]
        f = self.forces
        if os.environ.get('ANA_BIDIR', '0') == '1':
            f = 2.0 * f - 1.0          # 双向：-1..1，能往回蹬
        target = self.q_stand + self.posture_offset * f
        tau = self.kp * (target - q) - self.kd * qd
        self.data.ctrl[:] = np.clip(tau, self.ctrl_lo, self.ctrl_hi)

    def step(self, forces=None) -> None:
        if forces is not None:
            arr = np.asarray(forces, np.float32).ravel()
        self.forces = arr[:12]
        self.torso = (float(arr[12]) if len(arr) > 12
                      else float(getattr(self, 'torso_cmd', 0.0)))
        for _ in range(self.substeps):
            self._pd_step()
            self.mj.mj_step(self.model, self.data)

    # ------------------------------------------------------------ 状态
    def state(self) -> dict:
        q = self.data.qpos[7:19]
        quat = self.data.qpos[3:7]           # w x y z
        w, x, y, z = quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        gyro = self.data.qvel[3:6]
        height = float(self.data.qpos[2])
        vel = self.data.qvel[0:3]
        return dict(height=height, roll=float(roll), pitch=float(pitch),
                    yaw=float(yaw), gyro=gyro.copy(), vel=vel.copy(),
                    q=q.copy(), qd=self.data.qvel[6:18].copy())

    def fallen(self) -> bool:
        s = self.state()
        return s["height"] < 0.16 or abs(s["roll"]) > 0.9 or abs(s["pitch"]) > 0.9

    # ------------------------------------------------------------ 感觉
    def observe(self, spec=None) -> Observation:
        s = self.state()
        out = Observation()
        if spec is None:
            return out
        # 本体：12 关节角（按限位归一）+ 12 角速度 + 12 力矩
        qn = (s["q"] - self.jnt_lo) / np.maximum(self.jnt_hi - self.jnt_lo, 1e-6)
        qdn = np.clip((s["qd"] + 20.0) / 40.0, 0, 1)
        taun = np.clip((self.data.ctrl + 45.0) / 90.0, 0, 1)
        pro = np.zeros(spec.proprio_ch, np.float32)
        n = min(12, spec.proprio_ch)
        pro[:n] = np.clip(qn[:n], 0, 1)
        if spec.proprio_ch >= 24:
            pro[12:24] = qdn
        if spec.proprio_ch >= 36:
            pro[24:36] = taun
        out.proprio = pro
        # 平衡：roll/pitch/yaw + 陀螺 + 高度 + 三轴线速度
        bal = np.zeros(spec.balance_ch, np.float32)
        vals = [0.5 + s["roll"] / 1.0, 0.5 + s["pitch"] / 1.0, 0.5 + s["yaw"] / 3.14,
                0.5 + s["gyro"][0] / 10.0, 0.5 + s["gyro"][1] / 10.0,
                0.5 + s["gyro"][2] / 10.0, np.clip(s["height"] / 0.4, 0, 1),
                0.5 + s["vel"][0] / 2.0, 0.5 + s["vel"][1] / 2.0, 0.5 + s["vel"][2] / 2.0]
        bal[: min(len(vals), spec.balance_ch)] = np.clip(vals, 0, 1)[:spec.balance_ch]
        out.balance = bal
        # 触地：四条腿的接触力（用足端与地面距离近似）
        touch = np.zeros(spec.touch_ch, np.float32)
        feet = self._feet_heights()
        contact = np.clip(1.0 - feet / 0.03, 0, 1)
        touch[: min(4, spec.touch_ch)] = contact[: min(4, spec.touch_ch)]
        out.touch = touch
        out.vision = np.zeros((spec.px, spec.px, 3), np.float32)
        out.audio = np.zeros((spec.audio_bands, spec.audio_frames), np.float32)
        return out

    def _feet_heights(self) -> np.ndarray:
        """四条小腿末端的离地高度（用 geom 的 xpos 近似）。"""
        out = []
        for name in ("FL", "FR", "RL", "RR"):
            gid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_GEOM, f"{name}_foot")
            if gid < 0:
                out.append(0.0)
            else:
                out.append(float(self.data.geom_xpos[gid][2]))
        return np.asarray(out)