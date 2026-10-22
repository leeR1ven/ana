# -*- coding: utf-8 -*-
"""主循环：感觉 → 编码 → 大脑走几小步 → 肌肉 → 身体 → 新感觉。

一拍（50 Hz，20 ms）做这些事，顺序固定：

    1. 编码当前感觉（温度计正反，每对恰好一个）
    2. 大脑走 spec.substeps 小步（每小步 dt = 1/(rate×substeps)）
    3. 读运动区：每块肌肉亮了几个力度神经元 → 出力档
    4. 把出力写回身体 / 仿真；身体产生下一拍的感觉
    5. 记录遥测（耗时、前额叶码、各层活跃数、肌肉力）

v0 没有记忆连接：前额叶每拍的活动只是"当前输入在组合检测器上的投影"，
不会自持。等 group 40 的联想边长出来，前额叶才会开始"想"。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tick:
    """一拍的观测快照。"""
    t: int
    forces: np.ndarray
    force_counts: np.ndarray
    pfc_active: int
    pfc_code: np.ndarray
    layer_active: dict
    ms: dict
    meta: dict = field(default_factory=dict)


@dataclass
class Trace:
    forces: list = field(default_factory=list)
    force_counts: list = field(default_factory=list)
    pfc_active: list = field(default_factory=list)
    pfc_code: list = field(default_factory=list)
    ms: list = field(default_factory=list)
    meta: list = field(default_factory=list)

    def add(self, tick: Tick) -> None:
        self.forces.append(tick.forces.copy())
        self.force_counts.append(tick.force_counts.copy())
        self.pfc_active.append(tick.pfc_active)
        self.pfc_code.append(tick.pfc_code.copy())
        self.ms.append(dict(tick.ms))
        self.meta.append(dict(tick.meta))

    def arrays(self) -> dict:
        return dict(
            forces=np.stack(self.forces) if self.forces else np.zeros((0, 0)),
            force_counts=np.stack(self.force_counts) if self.force_counts else np.zeros((0, 0)),
            pfc_active=np.asarray(self.pfc_active, np.int32),
            ms_total=np.asarray([m.get("total", 0.0) for m in self.ms], np.float32),
            ms_step=np.asarray([m.get("step", 0.0) for m in self.ms], np.float32),
            ms_encode=np.asarray([m.get("encode", 0.0) for m in self.ms], np.float32),
        )

    def save(self, path: str) -> None:
        a = self.arrays()
        np.savez_compressed(path, **a)
        # 前额叶码是变长稀疏编号，单独存成 object 数组
        np.savez_compressed(path.replace(".npz", "_pfc.npz"),
                            pfc=np.array(self.pfc_code, dtype=object), allow_pickle=True)


class Cortex:
    """把 Brain 和身体接起来的最小闭环。"""

    def __init__(self, brain, body=None, trace: bool = True, plastic=None):
        from .plastic import PlasticRule
        self.brain = brain
        self.engine = brain.engine
        self.spec = brain.spec
        self.body = body
        self.trace = Trace() if trace else None
        self.plastic = plastic if plastic is not None else PlasticRule()
        self.last_plastic = dict(edges=0, grown=0, strengthened=0, pruned=0)
        self.t = 0

    # ------------------------------------------------------------ 一拍
    def tick(self, obs, record: bool = True) -> Tick:
        sp = self.spec
        t0 = time.perf_counter()
        buf = self.brain.encoder.encode(obs)
        self.engine.clear_external()
        ring = getattr(self.brain, "time_ring", None)
        if ring is not None:
            ring.step(self.engine)          # 海马体：每拍恰好一个时间神经元亮
        self.engine.set_input(buf)
        t1 = time.perf_counter()
        for _ in range(max(1, int(sp.substeps))):
            self.engine.step()
        binder = getattr(self.brain, "binder", None)
        if binder is not None:
            binder.step(self.brain.pfc_code())
            if binder.t % max(1, int(sp.hippo_flush_every)) == 0:
                binder.flush()
        t2 = time.perf_counter()
        forces = self.brain.forces()
        counts = self.brain.force_counts()
        need_code = (self.trace is not None) or bool(
            self.plastic is not None and self.plastic.cfg.enabled)
        code = self.brain.pfc_code() if need_code else np.zeros(0, np.int32)
        # v0 这里一定返回 0：plastic 默认关闭。训练阶段打开，靠"先亮→后亮"
        # 长前额叶内部的有向联想边（group 40）。
        if self.plastic is not None:
            self.last_plastic = self.plastic.apply(self.engine, code)
        t3 = time.perf_counter()
        pfc_active = int(len(code)) if need_code else self.brain.pfc_active()
        tk = Tick(t=self.t, forces=forces, force_counts=counts,
                  pfc_active=pfc_active, pfc_code=code,
                  layer_active={},
                  ms=dict(encode=(t1 - t0) * 1e3, step=(t2 - t1) * 1e3,
                          read=(t3 - t2) * 1e3, total=(t3 - t0) * 1e3),
                  meta=dict(obs.meta))
        if record and self.trace is not None:
            self.trace.add(tk)
        self.t += 1
        return tk

    # ------------------------------------------------------------ 跑一段
    def run(self, ticks: int, body=None, log_every: int = 0):
        """没有 body 时用上一次观测重复喂（检查通路/计时）；有 body 时闭环。"""
        body = body or self.body
        obs = getattr(self, "_last_obs", None)
        if obs is None:
            from .world import sample_observation
            obs = sample_observation(self.spec, np.random.default_rng(0), 0)
        for i in range(ticks):
            if body is not None:
                if i > 0:
                    body.step(self._last_forces)
                obs = body.observe()
            tk = self.tick(obs)
            self._last_forces = tk.forces
            self._last_obs = obs
            if log_every and (i % log_every == 0):
                print(f"  t={i:5d}  pfc={tk.pfc_active:5d}  "
                      f"force={np.round(tk.forces, 2)}  {tk.ms['total']:.2f} ms")
        return self.trace

    # ------------------------------------------------------------ 计时
    def bench(self, ticks: int = 30) -> dict:
        from .world import sample_observation
        rng = np.random.default_rng(5)
        obs = sample_observation(self.spec, rng, 0)
        for _ in range(3):
            self.tick(obs, record=False)
        ts = []
        for i in range(ticks):
            t0 = time.perf_counter()
            self.engine.set_input(self.brain.encoder.encode(obs))
            for _ in range(max(1, int(self.spec.substeps))):
                self.engine.step()
            _ = self.brain.forces()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.asarray(ts)
        return dict(mean_ms=float(ts.mean()), p95_ms=float(np.percentile(ts, 95)),
                    max_ms=float(ts.max()), ticks=ticks,
                    budget_ms=1000.0 / self.spec.rate_hz)
