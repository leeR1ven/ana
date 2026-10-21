# -*- coding: utf-8 -*-
"""ANA 规格：规模、几何、层计划。

这里只放"架构上是什么"，不放"怎么算"。所有数字都能被改，改完 `validate()`
会告诉你哪里在架构上不成立（越界、区重叠、层比例不递增……）。

设计来源：
  · 每层活跃数恒定（用户的"十盏灯"要求）  -> 精确 top-k
  · 每层 = 通道数 × 递增比例，越靠近前额叶越多  -> layer_ratio
  · 每个区在 (y,z) 平面占一条不重叠的带，沿 x 由浅到深     -> Geometry
  · 传送前额叶：1:1 专用线 + 两级中继                     -> pfc_fan_in / pfc_ratio
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# ---------------------------------------------------------------- 区与条带
# 顺序即 y 轴由小到大的顺序；条带互不重叠，全部在 [0,1]。
BANDS: dict[str, tuple[float, float]] = {
    "vision": (0.000, 0.300),
    "audio": (0.305, 0.400),
    "touch": (0.405, 0.500),
    "proprio": (0.505, 0.620),
    "balance": (0.625, 0.720),
}
MOTOR_BAND = (0.880, 0.960)

REGIONS = ("vision", "audio", "touch", "proprio", "balance")


@dataclass
class Geometry:
    """各区的 y/z 条带，以及沿 x 的深度。x 只表示"离前额叶多远"，不参与就近。"""

    bands: dict = field(default_factory=lambda: dict(BANDS))
    motor_band: tuple = MOTOR_BAND
    x_input: float = 0.00
    x_l1: float = 0.08
    x_layer_step: float = 0.06     # L1..L5 的 x 间隔
    x_pfc: float = 0.60
    x_return: float = 0.46         # 回程中继的 x
    x_motor_relay: float = 0.70
    x_motor_gate: float = 0.78
    x_force: float = 0.82

    def layer_x(self, depth: int) -> float:
        """depth=1 是 L1；2..5 每深一层 +x_layer_step。"""
        return self.x_l1 + self.x_layer_step * (depth - 1)


@dataclass
class Spec:
    # ---- 视觉 ----
    px: int = 32                    # 视觉边长（px×px×3 通道）
    n_level: int = 10               # 每通道温度计档数（= 每对正反里的档数）
    # ---- 其它感觉 ----
    audio_bands: int = 32
    audio_frames: int = 4
    touch_ch: int = 20              # 4 足 × 3 轴 + 8 处体表
    proprio_ch: int = 36            # 12 关节 × (角度, 角速度) + 12 关节力矩
    balance_ch: int = 10           # roll/pitch/yaw + 三轴陀螺 + 高度 + 三轴线速度

    # ---- 区内链 ----
    layer_ratio: tuple = (2.0, 3.0, 4.0, 5.0)   # L2..L5 = 通道数 × 比例（必须递增）
    k_near: int = 8                             # 每个神经元接上一层几个
    l1_active: float = 0.50                     # L1 活跃比例（ON/OFF 各半）
    layer_active: tuple = (0.40, 0.30, 0.20, 0.10)  # L2..L5 的 top-k 比例
    # ---- 前额叶 ----
    pfc_fan_in: int = 1             # 一个前额叶神经元收几条专用线（1 = 1:1，每个神经元一个"字"）
    pfc_ratio: float = 1.0          # 前额叶组大小 = 该层神经元数 × 这个比例
    pfc_thr_frac: float = 0.15      # 门槛 = 入边权重和 × 这个比例（1:1 时≈直通）
    pfc_soft_frac: float = 0.35
    pfc_target_frac: float = 0.10   # 整体阈值调控的目标活跃比例（1:1 时≈该层活跃率）
    # 常亮/常灭不往前额叶送（按跨输入 std 筛，不按当前亮不亮）
    pfc_filter: bool = True
    filter_min_rate: float = 0.02   # 参考分布里活跃率的下限（低于=常灭）
    filter_max_rate: float = 0.60   # 上限（高于=常亮；这种单元几乎不带"是哪一个输入"）
    filter_ticks: int = 12          # 用多少个参考场景统计
    # 前额叶动态抑制：每拍只留最强的这个比例（0 = 关）。
    # 字神经元网络的"动态抑制"；v0 默认关，P1 对照实验用。
    pfc_topk: float = 0.05
    # 前额叶自适应（习惯化）：每拍减掉自己的滑动平均驱动。
    # 0 = 关；0.05 相当于约 20 小步（200 ms）的时间常数。
    pfc_adapt: float = 0.0
    motor_group: int = 512          # 运动相关前额叶组大小（跨区汇聚）
    motor_target_frac: float = 0.10  # 运动组标定目标活跃比例（比前额叶松）
    # ---- 运动区 ----
    n_muscle: int = 12              # 12 个关节（以后接 Go2）
    n_force: int = 10               # 每块肌肉力度档数
    # ---- 时钟 ----
    rate_hz: float = 50.0           # 控制拍频
    substeps: int = 2               # 每拍走几步（越小越省、越大越快穿透深链）
    # ---- 其它 ----
    seed: int = 7

    # ------------------------------------------------ 通道数 / 层计划
    def channels(self) -> dict[str, int]:
        ch = {
            "vision": self.px * self.px * 3,
            "audio": self.audio_bands * self.audio_frames,
            "touch": self.touch_ch,
            "proprio": self.proprio_ch,
            "balance": self.balance_ch,
        }
        return ch

    def active_regions(self) -> tuple:
        return tuple(r for r in REGIONS if r in self.channels())

    def layers(self) -> list[dict]:
        """每层：区、深度(1..5)、神经元数、top-k 个数。"""
        out = []
        for r, c in self.channels().items():
            out.append(dict(region=r, depth=1, size=2 * c,
                            k=int(round(self.l1_active * 2 * c))))
            for i, ratio in enumerate(self.layer_ratio):
                size = int(round(ratio * c))
                k = max(1, int(round(self.layer_active[i] * size)))
                out.append(dict(region=r, depth=i + 2, size=size, k=k))
        return out

    def topk_layers(self) -> list[dict]:
        return list(self.layers())

    # ------------------------------------------------ 规模估计
    def estimate(self) -> dict:
        layers = self.layers()
        n_layer = sum(L["size"] for L in layers if L["depth"] > 0)
        n_in = sum(2 * self.n_level * c for r, c in self.channels().items())
        # 区内前向：L1 由输入直接连（每对 N 条），L2..L5 各 k_near 条
        fwd = 0
        for r, c in self.channels().items():
            fwd += 2 * c * self.n_level
            for L in layers:
                if L["region"] == r and L["depth"] >= 2:
                    fwd += self.k_near * L["size"]
        n_relay = 2 * n_layer                       # 两级专用线中继
        pfc = int(round(self.pfc_ratio * n_layer))
        pfc_in = self.pfc_fan_in * pfc
        ret = 3 * n_layer + 64 * len([L for L in layers if L["depth"] >= 2])
        motor = 3 * self.motor_group + 6 * self.n_muscle * self.n_force
        return dict(
            neurons=n_in + n_layer + n_relay + pfc + int(round(1.5 * n_layer)) + motor,
            edges=fwd + 2 * n_layer + pfc_in + ret + motor,
            input_cells=n_in, layer_neurons=n_layer, relays=n_relay,
            pfc=pfc, forward=fwd, pfc_in=pfc_in, return_edges=ret, motor_edges=motor,
        )

    # ------------------------------------------------ 校验
    def validate(self) -> list[str]:
        e: list[str] = []
        for nm in ("px", "n_level", "audio_bands", "audio_frames", "touch_ch",
                   "proprio_ch", "balance_ch", "k_near", "n_muscle", "n_force",
                   "motor_group"):
            if int(getattr(self, nm)) < 1:
                e.append(f"{nm} 必须 ≥ 1")
        if self.n_level < 2:
            e.append("n_level 至少 2（否则温度计退化成二值）")
        if len(self.layer_ratio) < 1:
            e.append("layer_ratio 至少 1 个数")
        if any(b <= a for a, b in zip(self.layer_ratio, self.layer_ratio[1:])):
            e.append("layer_ratio 必须递增（越靠近前额叶越多）")
        if len(self.layer_active) != len(self.layer_ratio):
            e.append("layer_active 长度必须等于 layer_ratio")
        if any(not (0.0 < a < 1.0) for a in self.layer_active):
            e.append("layer_active 必须在 (0,1)")
        if not (1 <= self.pfc_fan_in <= 64):
            e.append("pfc_fan_in 必须在 1..64")
        if not (0.0 < self.pfc_target_frac < 1.0):
            e.append("pfc_target_frac 必须在 (0,1)")
        if not (0.0 <= self.filter_min_rate < self.filter_max_rate <= 1.0):
            e.append("filter_min_rate 必须 < filter_max_rate，且在 [0,1]")
        if int(self.filter_ticks) < 1:
            e.append("filter_ticks 至少 1")
        if not (0.0 <= self.pfc_topk < 1.0):
            e.append("pfc_topk 必须在 [0,1)")
        if not (0.0 <= self.pfc_adapt < 1.0):
            e.append("pfc_adapt 必须在 [0,1)")

        if not (1 <= int(self.substeps) <= 8):
            e.append("substeps 必须在 1..8")
        if self.rate_hz <= 0:
            e.append("rate_hz 必须 > 0")
        return e


@dataclass
class ScalePreset:
    """常用规模。full 是默认；large 会明显吃满 CPU。"""

    name: str
    px: int
    n_level: int
    pfc_ratio: float = 1.0
    audio_bands: int = 32
    audio_frames: int = 4

    def spec(self, **kw) -> Spec:
        return replace(Spec(), px=self.px, n_level=self.n_level,
                       pfc_ratio=self.pfc_ratio, audio_bands=self.audio_bands,
                       audio_frames=self.audio_frames, **kw)


# 实测（i5-12600KF / torch CPU / 2 小步每拍）：32×32 → 约 8 ms/拍；
# 36×36 → 约 10 ms/拍；40×40 → 约 12 ms/拍。20 ms 预算下默认取 xl。
PRESETS = {
    "tiny": ScalePreset("tiny", 8, 8, 1.0, 12, 2),
    "small": ScalePreset("small", 16, 10, 1.0, 16, 2),
    "full": ScalePreset("full", 32, 10, 1.0, 32, 4),
    "xl": ScalePreset("xl", 36, 10, 1.0, 32, 4),
    "large": ScalePreset("large", 40, 10, 1.0, 32, 4),
}


def build_spec(name: str = "xl", **kw) -> Spec:
    """按预设造 Spec，并可覆盖任意字段。"""
    if name not in PRESETS:
        raise KeyError(f"未知预设 {name}；可选 {sorted(PRESETS)}")
    sp = PRESETS[name].spec(**kw)
    errs = sp.validate()
    if errs:
        raise ValueError("Spec 不合法：\n  - " + "\n  - ".join(errs))
    return sp


def bands_ordered(geo: Geometry | None = None) -> list[tuple[float, float, str]]:
    g = geo or Geometry()
    return sorted((a, b, r) for r, (a, b) in g.bands.items())


def validate_geometry(sp: Spec, geo: Geometry) -> list[str]:
    e: list[str] = []
    for r in sp.active_regions():
        if r not in geo.bands:
            e.append(f"Geometry.bands 缺区 {r}")
            continue
        y0, y1 = geo.bands[r]
        if not (0.0 <= y0 < y1 <= 1.0):
            e.append(f"{r} 的条带 ({y0},{y1}) 必须满足 0 ≤ y0 < y1 ≤ 1")
    spans = [(a, b, r) for a, b, r in bands_ordered(geo) if r in sp.channels()]
    for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
        if b0 < a1:
            e.append(f"区 {an} 与 {bn} 的条带重叠：[{a0},{a1}] / [{b0},{b1}]")
    my0, my1 = geo.motor_band
    if spans and my0 < spans[-1][1]:
        e.append(f"运动区条带 ({my0},{my1}) 与 {spans[-1][2]} 重叠")
    if not (geo.x_input < geo.x_l1 < geo.layer_x(max(1, len(sp.layer_ratio) + 1))
            < geo.x_return < geo.x_pfc < geo.x_motor_relay < geo.x_motor_gate
            < geo.x_force):
        e.append("x 顺序必须是：输入 < 各层 < 回程 < 前额叶 < 运动中继 < 门控 < 力度")
    return e


def band_positions(geo: Geometry, region: str, n: int, rng: np.random.Generator,
                   x: float) -> np.ndarray:
    """在区条带里均匀撒 n 个位置（y 在带内，z 在 [0,1]）。"""
    y0, y1 = geo.bands[region]
    u = (np.arange(n, dtype=np.float32) + 0.5) / n
    y = y0 + u * (y1 - y0)
    z = rng.random(n).astype(np.float32)
    p = np.empty((n, 3), np.float32)
    p[:, 0] = x
    p[:, 1] = y
    p[:, 2] = z
    return p
