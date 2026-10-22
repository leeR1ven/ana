# -*- coding: utf-8 -*-
"""按规格把大脑搭出来。

一次 build 的产物（`Brain`）就是 v0 的完整空框架：

    感觉细胞（温度计正反） → L1 ON/OFF → L2..L5（就近多输入 + 精确 top-k）
        │  每一层都送：1:1 专用线 → 两级中继 → 该层的前额叶组
        ↓
    前额叶（汇聚端：多输入 + 门槛 = 组合检测；内部连接数 = 0）
        ↓  回程：前额叶 → 两级回程中继 → 按位置送回本层（只做调制，不落 L1）
    运动区：运动相关前额叶组 → 两级运动中继 → 肌肉力度档 → 抑制 I / 去抑制 D 门控

**这一版没有的东西（故意的）**：
  · 前额叶内部连接（group 40）——0 条。记忆/联想以后靠它长出来。
  · 任何可塑边（group -1）——0 条。
  · 结构增删（grow_units 接口留着，默认不用）。

几条几何约定：
  · 区在 y 上占一条不重叠的带；z 是带内位置；x 只是"离前额叶多远"。
  · 每层的神经元按"像素优先"排：i = 像素 p × 每像素槽位 R + 槽位 j。
    这样"就近"可以用 (y,z) 上的 3×3 像素邻域算，不用全对全距离。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .encoding import Encoder
from .engine import Engine, EngineConfig
from .spec import (Geometry, Spec, band_positions, build_spec, validate_geometry)
from .world import sample_observation


class G:
    """边的分组。不要复用这些号：结构/统计代码靠它们找通路。"""
    FWD = 0            # 区内前向：输入→L1、L2..L5
    MOTOR_NON = 2      # 运动无关前额叶 → I
    MOTOR_GRP = 3      # 运动相关组 → D（去抑制）
    MOTOR_DEINH = 4    # D → I（抑制）
    MOTOR_DESC = 5     # 运动组 → 运动中继（下降链）
    MOTOR_DRIVE = 6    # 运动中继 → 肌肉力度
    MOTOR_BLOCK = 7    # I → 力度（压住）
    MOTOR_FB = 15      # 力度 → I（自反馈，预留）
    RET_FROM_PFC = 19  # 前额叶 → 回程抑制细胞
    RET_INH = 16       # 回程抑制细胞 → 第一级
    RET_L1 = 17        # 前额叶 → 第一级（随机铺开）
    RET_L2 = 18        # 第二级 → 本层神经元
    REGION_LINK = 55   # 区与区之间的直连（出生固定、不可改）
    REFLEX_INH = 51    # 抑制细胞 → 反射中继（前额叶用它压制反射）
    PFC = 30           # 往前额叶送的整条通路（层→中继→中继→额叶）
    PLASTIC = -1       # 预留：可塑边（v0 为 0 条）
    PL_PFC = 40        # 预留：前额叶内部联想边（v0 为 0 条）


# 按深度定门槛/带宽比例：每深一层活跃密度约减半，比例必须跟着减半
DEPTH_FRAC = {1: (0.25, 0.45), 2: (0.25, 0.45), 3: (0.08, 0.30),
              4: (0.04, 0.25), 5: (0.02, 0.18)}


@dataclass
class Geo2D:
    """一层的二维排布：H×W 个像素，每个像素 R 个槽位。"""
    H: int
    W: int
    R: int
    pix: np.ndarray

    @property
    def n_pix(self) -> int:
        return self.H * self.W


@dataclass
class LayerInfo:
    name: str
    region: str
    depth: int
    ids: np.ndarray
    k: int

    @property
    def size(self) -> int:
        return int(len(self.ids))


@dataclass
class Brain:
    engine: Engine
    spec: Spec
    geo: Geometry
    ids: dict
    layers: dict
    pfc_ids: np.ndarray
    pfc_groups: dict
    motor: dict
    encoder: Encoder
    filtered: dict = field(default_factory=dict)   # 层名 → 该层被"常亮/常灭"筛掉的掩码

    # ---------------------------------------------------------- 读
    def forces(self) -> np.ndarray:
        """每块肌肉的出力档（0..1）= 该肌肉组里亮着的力度神经元比例。"""
        return self.force_counts().astype(np.float32) / float(self.spec.n_force)

    def force_counts(self) -> np.ndarray:
        a = self.engine.act_np
        return np.array([int((a[self.motor["force"][i]] > 0.35).sum())
                         for i in range(self.spec.n_muscle)], np.int16)

    def pfc_code(self, thr: float = 0.35) -> np.ndarray:
        a = self.engine.act_np
        return self.pfc_ids[a[self.pfc_ids] > thr].astype(np.int32)

    def pfc_active(self, thr: float = 0.35) -> int:
        return int((self.engine.act_np[self.pfc_ids] > thr).sum())

    def layer_active(self, name: str, thr: float = 0.35) -> int:
        L = self.layers[name]
        return int((self.engine.act_np[L.ids] > thr).sum())

    def region_code(self, region: str, depth: int = 5, thr: float = 0.35) -> np.ndarray:
        L = self.layers[f"{region}.L{depth}"]
        return L.ids[self.engine.act_np[L.ids] > thr]

    def summary(self) -> dict:
        est = self.spec.estimate()
        kept = (sum(int((~m).sum()) for m in self.filtered.values())
                if self.filtered else len(self.pfc_ids))
        return dict(
            neurons=self.engine.n, edges=self.engine.n_edges,
            layers=len(self.layers), pfc=kept,
            pfc_total=int(len(self.pfc_ids)),
            filtered_pfc=int(len(self.pfc_ids) - kept),
            input_cells=self.encoder.n_in,
            plastic_edges=int((self.engine.group == G.PLASTIC).sum()),
            pfc_internal=int((self.engine.group == G.PL_PFC).sum()),
            spec_estimate=est,
        )


# ---------------------------------------------------------------- 小工具
def _grid_for(region: str, spec: Spec) -> tuple[int, int, int]:
    """返回 (H, W, 每像素通道数)。H×W 是"像素"数，用于就近。"""
    if region == "vision":
        return spec.px, spec.px, 3
    if region == "audio":
        return spec.audio_bands, spec.audio_frames, 1
    if region == "touch":
        return 4, max(1, spec.touch_ch // 4), 1
    if region == "proprio":
        h = int(round(np.sqrt(spec.proprio_ch)))
        while spec.proprio_ch % h:
            h -= 1
        return h, spec.proprio_ch // h, 1
    if region == "balance":
        return 2, max(1, spec.balance_ch // 2), 1
    if region == "self":
        return 3, max(1, spec.n_muscle // 3), 1
    raise KeyError(region)


def _pixel_positions(geo: Geometry, region: str, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """每个像素的 (y, z) 中心。"""
    y0, y1 = geo.bands[region]
    r = np.arange(H * W) // W
    c = np.arange(H * W) % W
    y = (y0 + (r + 0.5) / H * (y1 - y0)).astype(np.float32)
    z = ((c + 0.5) / W).astype(np.float32)
    return y, z


def _layer_positions(geo: Geometry, region: str, H: int, W: int, R: int,
                     x: float, rng: np.random.Generator) -> np.ndarray:
    y0, z0 = _pixel_positions(geo, region, H, W)
    y0 = np.repeat(y0, R)
    z0 = np.repeat(z0, R)
    slot = np.tile(np.arange(R), H * W)
    span = (geo.bands[region][1] - geo.bands[region][0])
    y = y0 + (slot - (R - 1) / 2.0) * (0.02 * span / max(1, R))
    z = z0 + (slot - (R - 1) / 2.0) * (0.02 / max(1, W * R))
    p = np.empty((len(y), 3), np.float32)
    p[:, 0] = x
    p[:, 1] = y.astype(np.float32)
    p[:, 2] = np.clip(z, 0.0, 1.0).astype(np.float32)
    return p


def _band_pos(geo: Geometry, region: str, n: int, rng: np.random.Generator,
              x: float) -> np.ndarray:
    """在区条带里撒位置；"motor" 用运动区条带。"""
    if region in geo.bands:
        return band_positions(geo, region, n, rng, x)
    y0, y1 = geo.motor_band
    u = (np.arange(n, dtype=np.float32) + 0.5) / max(1, n)
    y = y0 + u * (y1 - y0)
    z = rng.random(n).astype(np.float32)
    p = np.empty((n, 3), np.float32)
    p[:, 0] = x
    p[:, 1] = y
    p[:, 2] = z
    return p


def _sample_no_replace(rng: np.random.Generator, pool: np.ndarray,
                       n_rows: int, k: int) -> np.ndarray:
    """从 pool 里给 n_rows 行各抽 k 个（不重复）。pool 一般远大于 k。"""
    k = int(min(k, len(pool)))
    if k <= 0 or n_rows <= 0:
        return np.zeros((max(0, n_rows), 0), np.int64)
    idx = rng.integers(0, len(pool), size=(n_rows, k))
    # 去重：碰撞率很低；撞了的行用"不放回抽样"重抽，保证每行 k 个都不重复
    bad = np.array([len(set(row)) < k for row in idx])
    if bad.any() and k <= len(pool):
        for i in np.flatnonzero(bad):
            idx[i] = rng.choice(len(pool), size=k, replace=False)
    return pool[idx]


def connect_near(eng: Engine, dst: np.ndarray, src: np.ndarray, *,
                 src_geo: Geo2D, dst_geo: Geo2D, k: int,
                 w_range=(0.05, 0.20), group: int = G.FWD,
                 thr_frac: float | None = None, soft_frac: float | None = None,
                 rng: np.random.Generator, bias: int = 3,
                 window: int = 1) -> int:
    """就近多输入：每个目标只从 (y,z) 上 3×3 像素邻域里的上一层神经元中挑 k 个。

    参考实现是"取 3k 个最近候选，再随机挑 k"，这里保留同样语义（bias=3）。
    """
    if len(dst) == 0 or len(src) == 0:
        return 0
    k = int(min(k, src_geo.R * (2 * window + 1) ** 2))
    if k <= 0:
        return 0
    pos = eng.pos
    src_ids2 = src.reshape(src_geo.n_pix, src_geo.R)
    src_pos2 = pos[src].reshape(src_geo.n_pix, src_geo.R, 3)
    rows = dst_geo.pix // dst_geo.W
    cols = dst_geo.pix % dst_geo.W
    H, W = dst_geo.H, dst_geo.W
    offs = [(dy, dx) for dy in range(-window, window + 1)
            for dx in range(-window, window + 1)]
    cand_ids, cand_pos = [], []
    for dy, dx in offs:
        r = rows + dy
        c = cols + dx
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        p = np.where(ok, r * W + c, 0)
        ids = src_ids2[p]
        ps = src_pos2[p]
        cand_ids.append(np.where(ok[:, None], ids, -1))
        cand_pos.append(ps)
    cand_ids = np.concatenate(cand_ids, axis=1)          # (n, 9R)
    cand_pos = np.concatenate(cand_pos, axis=1)          # (n, 9R, 3)
    dpos = pos[dst][:, None, 1:3]
    d = np.linalg.norm(cand_pos[:, :, 1:3] - dpos, axis=2)
    d[cand_ids < 0] = np.inf
    pool = min(bias * k, cand_ids.shape[1])
    sel = np.argpartition(d, pool - 1, axis=1)[:, :pool]
    order = np.argsort(rng.random(sel.shape), axis=1)
    sel = np.take_along_axis(sel, order, axis=1)[:, :k]
    chosen = np.take_along_axis(cand_ids, sel, axis=1)
    good = chosen >= 0
    if not good.any():
        return 0
    dst_rep = np.repeat(dst, k).reshape(len(dst), k)
    n = eng.connect(dst_rep[good], chosen[good],
                    rng.uniform(*w_range, size=int(good.sum())), group=group)
    if thr_frac is not None or soft_frac is not None:
        eng.set_thr(dst, thr_frac=thr_frac, soft_frac=soft_frac)
    return n


# ---------------------------------------------------------------- 主流程
def build(spec: Spec | None = None, engine_cfg: EngineConfig | None = None,
          geo: Geometry | None = None, verbose: bool = True,
          do_calibrate: bool = True, calib_ticks: int = 8) -> Brain:
    spec = spec or build_spec("xl")
    geo = geo or Geometry()
    errs = spec.validate() + validate_geometry(spec, geo)
    if errs:
        raise ValueError("规格不合法：\n  - " + "\n  - ".join(errs))
    rng = np.random.default_rng(spec.seed)
    cfg = engine_cfg or EngineConfig(seed=spec.seed,
                                      dt=1.0 / (spec.rate_hz * max(1, spec.substeps)))
    eng = Engine(cfg)
    N = spec.n_level
    ch = spec.channels()

    ids: dict = {"input": {}, "layers": {}, "relay": {}, "pfc": {}, "ret": {},
                 "motor": {}}
    layout: dict[str, tuple[int, int]] = {}
    layers: dict[str, LayerInfo] = {}
    pfc_groups: dict[tuple[str, int], np.ndarray] = {}
    geos: dict[str, Geo2D] = {}

    # ---------------------------------------------------------- 1. 感觉细胞
    for r in spec.active_regions():
        C = ch[r]
        off = eng.n
        H, W, cpp = _grid_for(r, spec)
        assert H * W * cpp == C, f"{r}: {H}×{W}×{cpp} != {C}"
        py, pz = _pixel_positions(geo, r, H, W)
        cpix = np.arange(C) // cpp
        # 每个通道：N 个 pos + N 个 neg
        lvl = np.tile(np.arange(2 * N, dtype=np.float32) % N, C)
        pair = np.tile(((np.arange(2 * N) // N) % 2), C).astype(np.float32)
        y = np.repeat(py[cpix], 2 * N) + (pair - 0.5) * 1e-4
        z = np.repeat(pz[cpix], 2 * N) + (lvl - N / 2) * 1e-5
        pos = np.stack([np.full(len(y), geo.x_input, np.float32),
                        y.astype(np.float32), z.astype(np.float32)], axis=1)
        block = eng.add_block(2 * C * N, kind="input", region=r, thr=0.0, soft=1.0,
                              counted=False, pos=pos)
        layout[r] = (off, off + 2 * C * N)
        ids["input"][r] = block.reshape(C, 2, N)

    n_in = eng.n
    eng.input_size = n_in

    # ---------------------------------------------------------- 2. 区内链
    bias_plan: list = []          # 出生偏置：等 finalize() 之后数组才有大小
    for r in spec.active_regions():
        C = ch[r]
        H, W, cpp = _grid_for(r, spec)
        n_pix = H * W
        reg_ids = ids["input"][r]
        # L1：每通道 ON/OFF 两格
        R1 = 2 * cpp
        size1 = 2 * C
        l1_geo = Geo2D(H, W, R1, np.arange(size1) // R1)
        l1 = eng.add_block(size1, kind="l1", region=r, depth=1, thr=0.0, soft=1.0,
                           pos=_layer_positions(geo, r, H, W, R1, geo.layer_x(1), rng))
        on_ids = l1[0::2]
        off_ids = l1[1::2]
        pos_cells = reg_ids[:, 0, :]          # (C, N)
        neg_cells = reg_ids[:, 1, :]
        eng.connect(np.repeat(on_ids, N), pos_cells.reshape(-1),
                    rng.uniform(0.6, 1.0, C * N), group=G.FWD)
        eng.connect(np.repeat(off_ids, N), neg_cells.reshape(-1),
                    rng.uniform(0.6, 1.0, C * N), group=G.FWD)
        k1 = max(1, int(round(spec.l1_active * size1)))
        # 层内不设绝对门槛：用"驱动 ÷ 自己的入边和"当相对强度，再由 top-k 选。
        # 这样不会因为"门槛刚好卡在活动水平上"让整层抖动（local3d 的做法）。
        # 归一化到"典型驱动 = 1"：入边和 × 上一层通常的活跃比例。
        # 输入层每通道恰好一半亮，所以比例是 0.5。
        eng.set_thr(l1, thr=0.0, soft=eng.in_weight_sum(l1) * 0.5)
        eng.register_topk(l1, k1)
        layers[f"{r}.L1"] = LayerInfo(f"{r}.L1", r, 1, l1, k1)
        geos[r] = l1_geo
        prev, prev_geo, prev_k, prev_size = l1, l1_geo, k1, size1
        # L2..L5
        for i, ratio in enumerate(spec.layer_ratio):
            depth = i + 2
            size = int(round(ratio * C))
            R = size // n_pix
            if R * n_pix != size:      # 保证像素优先排布
                raise ValueError(f"{r} L{depth}: size {size} 不能被像素数 {n_pix} 整除")
            g = Geo2D(H, W, R, np.arange(size) // R)
            blk = eng.add_block(size, kind="layer", region=r, depth=depth,
                                thr=0.0, soft=1.0,
                                pos=_layer_positions(geo, r, H, W, R,
                                                     geo.layer_x(depth), rng))
            if spec.bias_scale > 0:
                bias_plan.append((blk, rng.uniform(0.0, spec.bias_scale,
                                                   len(blk)).astype(np.float32)))
            connect_near(eng, blk, prev, src_geo=prev_geo, dst_geo=g,
                         k=spec.k_near, rng=rng, group=G.FWD)
            if spec.self_copy:
                # 自身拷贝（local3d 的做法）：同一像素在上一层的那一个神经元，
                # 1:1 拉一条权重 1.0 的线。让"同一个东西"在下一层仍有对应位置。
                same = np.minimum(g.pix * prev_geo.R, len(prev) - 1)
                eng.connect(blk, prev[same], np.ones(len(blk), np.float32), group=G.FWD)
            frac = max(1e-3, prev_k / max(1, prev_size))
            eng.set_thr(blk, thr=0.0, soft=eng.in_weight_sum(blk) * frac)
            k = max(1, int(round(spec.layer_active[i] * size)))
            eng.register_topk(blk, k)
            layers[f"{r}.L{depth}"] = LayerInfo(f"{r}.L{depth}", r, depth, blk, k)
            prev, prev_geo, prev_k, prev_size = blk, g, k, size
        ids["layers"][r] = [layers[f"{r}.L{d}"].ids for d in range(1, len(spec.layer_ratio) + 2)]

    # ---------------------------------------------------------- 3. 专用线 → 前额叶
    for name, L in layers.items():
        size = L.size
        x1 = geo.layer_x(L.depth) + (geo.x_pfc - geo.layer_x(L.depth)) / 3.0
        x2 = geo.layer_x(L.depth) + 2 * (geo.x_pfc - geo.layer_x(L.depth)) / 3.0
        src_pos = eng.pos[L.ids]
        r1 = eng.add_block(size, kind="relay", region=L.region, depth=L.depth,
                           thr=0.0, soft=1.0,
                           pos=np.stack([np.full(size, x1, np.float32), src_pos[:, 1],
                                         src_pos[:, 2]], axis=1))
        r2 = eng.add_block(size, kind="relay", region=L.region, depth=L.depth,
                           thr=0.0, soft=1.0,
                           pos=np.stack([np.full(size, x2, np.float32), src_pos[:, 1],
                                         src_pos[:, 2]], axis=1))
        eng.link(L.ids, r1, rng.uniform(0.9, 1.0, size), group=G.PFC)
        eng.link(r1, r2, rng.uniform(0.9, 1.0, size), group=G.PFC)
        psize = max(1, int(round(spec.pfc_ratio * size)))
        pp = band_positions(geo, L.region, psize, rng, geo.x_pfc)
        pfc = eng.add_block(psize, kind="pfc", region=L.region, depth=L.depth,
                            thr=0.0, soft=1.0, counted=True, pos=pp)
        fan = int(min(spec.pfc_fan_in, size))
        if fan == 1 and psize == size:
            # 1:1 专用线 = 一一对应：第 i 个前额叶神经元接第 i 个中继。
            # 不能随机采样（会重复、会漏掉特征），否则"一个字"就不是那个字了。
            pick = r2.reshape(-1, 1)
            eng.connect(pfc, pick.reshape(-1), rng.uniform(0.9, 1.0, psize), group=G.PFC)
        else:
            pick = _sample_no_replace(rng, r2, psize, fan)      # (psize, fan)
            eng.connect(np.repeat(pfc, fan), pick.reshape(-1),
                        rng.uniform(0.05, 0.20, psize * fan), group=G.PFC)
        eng.set_thr(pfc, thr_frac=spec.pfc_thr_frac, soft_frac=spec.pfc_soft_frac)
        pfc_groups[(L.region, L.depth)] = pfc
        ids["relay"][(L.region, L.depth)] = (r1, r2)
        # ---- 回程（不落 L1；只做调制）
        if L.depth >= 2:
            nr1 = max(4, min(32, size // 2))
            ret_inh = eng.add_block(1, kind="ret_inh", region=L.region, depth=L.depth,
                                    inh=True, thr=0.0, soft=1.0,
                                    pos=band_positions(geo, L.region, 1, rng, geo.x_return))
            ret1 = eng.add_block(nr1, kind="ret_relay", region=L.region, depth=L.depth,
                                 thr=0.0, soft=1.0,
                                 pos=np.stack([np.full(nr1, geo.x_return + 0.02, np.float32),
                                               rng.uniform(*geo.bands[L.region], nr1),
                                               rng.random(nr1)], axis=1))
            ret2 = eng.add_block(max(1, size // 2), kind="ret_relay", region=L.region,
                                 depth=L.depth, thr=0.0, soft=1.0,
                                 pos=np.stack([np.full(max(1, size // 2),
                                                       geo.x_return - 0.02, np.float32),
                                               rng.uniform(*geo.bands[L.region],
                                                           max(1, size // 2)),
                                               rng.random(max(1, size // 2))], axis=1))
            # 回程由"抑制神经元规则"自己选：抑制细胞广撒网地读前额叶整体活动，
            # 整体越活跃压得越狠；只有局部那条特别强的回程线压不住、能过线。
            # 思考焦点一换，上一帧那条不再是最强，就被整体抑制压回去。
            k1 = min(8, len(pfc))
            pick1 = _sample_no_replace(rng, pfc, nr1, k1)
            eng.connect(np.repeat(ret1, k1), pick1.reshape(-1),
                        rng.uniform(0.20, 0.30, nr1 * k1), group=G.RET_L1)
            # 抑制细胞读"一大片"前额叶（广撒网），所以它的活动≈整体活跃程度
            n_broad = min(96, len(pfc))
            picki = _sample_no_replace(rng, pfc, 1, n_broad)
            eng.connect(np.repeat(ret_inh, n_broad), picki.reshape(-1),
                        rng.uniform(0.05, 0.08, n_broad), group=G.RET_FROM_PFC)
            eng.set_thr(ret_inh, thr_frac=0.55, soft_frac=0.45)
            eng.connect(ret1, np.full(nr1, ret_inh[0], np.int32),
                        np.full(nr1, 1.6, np.float32), group=G.RET_INH)
            pick2 = _sample_no_replace(rng, ret1, len(ret2), 4)
            eng.connect(np.repeat(ret2, 4), pick2.reshape(-1),
                        rng.uniform(0.15, 0.25, len(ret2) * 4), group=G.RET_L1)
            # 第二级 → 本层（每个第二级单元看 2 个本层神经元，权重小=调制）
            tgt = L.ids[rng.integers(0, size, size=(len(ret2), 2))]
            wret = rng.uniform(0.02, 0.06, size=(len(ret2), 2))
            eng.connect(tgt.reshape(-1), np.repeat(ret2, 2), wret.reshape(-1),
                        group=G.RET_L2)
            eng.set_thr(ret1, thr_frac=0.30, soft_frac=0.35)
            eng.set_thr(ret2, thr_frac=0.30, soft_frac=0.35)
            ids["ret"][(L.region, L.depth)] = (ret_inh, ret1, ret2)

    # ---------------------------------------------------------- 4. 运动区
    # 运动相关前额叶组 = 跨区汇聚的组合检测器；再按肌肉分成互不重叠的小组。
    # 一块肌肉的力度 = 它那一组里同时亮了几个（链式架构：亮 1 个→0 档，亮 8 个→满档）。
    pool = np.concatenate([ids["relay"][(layers[f"{r}.L5"].region, 5)][1]
                           for r in spec.active_regions() if f"{r}.L5" in layers])
    nm = spec.motor_group
    mpfc = eng.add_block(nm, kind="pfc", region="motor", depth=6, thr=0.0, soft=1.0,
                         counted=True,
                         pos=_band_pos(geo, "motor", nm, rng, geo.x_pfc))
    pick = _sample_no_replace(rng, pool, nm, 8)
    eng.connect(np.repeat(mpfc, 8), pick.reshape(-1),
                rng.uniform(0.05, 0.20, nm * 8), group=G.MOTOR_GRP)
    eng.set_thr(mpfc, thr_frac=spec.pfc_thr_frac, soft_frac=spec.pfc_soft_frac)
    mr1 = eng.add_block(nm, kind="motor_relay", region="motor", depth=6, thr=0.0, soft=1.0,
                        pos=_band_pos(geo, "motor", nm, rng, geo.x_motor_relay))
    eng.link(mpfc, mr1, rng.uniform(0.9, 1.0, nm), group=G.MOTOR_DESC)
    # 按肌肉切分（余数给前面的肌肉）
    sub = np.array_split(np.arange(nm), spec.n_muscle)
    mforce = {}
    for m in range(spec.n_muscle):
        members = mr1[sub[m]]
        f = eng.add_block(spec.n_force, kind="force", region="motor", depth=6,
                          thr=0.0, soft=1.0,
                          pos=_band_pos(geo, "motor", spec.n_force, rng, geo.x_force))
        # 每根边权重 1：drive = 这一组里亮着的个数
        K = len(members)
        eng.connect(np.repeat(f, K), np.tile(members, spec.n_force),
                    np.ones(spec.n_force * K, np.float32), group=G.MOTOR_DRIVE)
        # 第 l 档：亮到 1.5+0.65·l 个就点（亮 1 个不出力，亮 8 个满档）
        thr = (1.5 + 0.65 * np.arange(spec.n_force, dtype=np.float32))
        eng.set_thr(f, thr=thr, soft=np.full(spec.n_force, 0.6, np.float32))
        mforce[m] = f
    # 门控：I（抑制）/ D（去抑制）
    nonmotor = np.concatenate([v for v in pfc_groups.values()])
    inh = eng.add_block(16, kind="motor_inh", region="motor", depth=6, inh=True,
                        thr=0.0, soft=1.0,
                        pos=_band_pos(geo, "motor", 16, rng, geo.x_motor_gate))
    pi = _sample_no_replace(rng, nonmotor, 16, 6)
    eng.connect(np.repeat(inh, 6), pi.reshape(-1), rng.uniform(0.1, 0.2, 16 * 6),
                group=G.MOTOR_NON)
    touch5 = layers.get("touch.L5")
    if touch5 is not None:
        pt = _sample_no_replace(rng, touch5.ids, 16, 4)
        eng.connect(np.repeat(inh, 4), pt.reshape(-1), rng.uniform(0.1, 0.2, 16 * 4),
                    group=G.MOTOR_NON)
    eng.set_thr(inh, thr_frac=0.30, soft_frac=0.35)
    deinh = eng.add_block(16, kind="motor_deinh", region="motor", depth=6, inh=True,
                          thr=0.0, soft=1.0,
                          pos=_band_pos(geo, "motor", 16, rng, geo.x_motor_gate + 0.02))
    pd_ = _sample_no_replace(rng, mpfc, 16, 8)
    eng.connect(np.repeat(deinh, 8), pd_.reshape(-1), rng.uniform(0.1, 0.2, 16 * 8),
                group=G.MOTOR_GRP)
    eng.set_thr(deinh, thr_frac=0.55, soft_frac=0.40)
    eng.link(deinh, inh, np.full(16, 1.2, np.float32), group=G.MOTOR_DEINH)
    for m in range(spec.n_muscle):
        f = mforce[m]
        ti = rng.integers(0, 16, size=(spec.n_force, 2))
        eng.connect(np.repeat(f, 2), inh[ti].reshape(-1),
                    np.full(spec.n_force * 2, 1.1, np.float32), group=G.MOTOR_BLOCK)

    ids["motor"] = dict(pfc=mpfc, relay=mr1, force=mforce, inh=inh, deinh=deinh)
    motor = dict(mpfc=mpfc, mr1=mr1, force=mforce, inh=inh, deinh=deinh, sub=sub)

    # ---- 区与区之间的直连（出生固定、不参与学习：这类连接天生难改）----
    if spec.region_links > 0:
        order = list(spec.active_regions())
        for a, b in zip(order, order[1:]):
            for src_r, dst_r in ((a, b), (b, a)):
                if f"{src_r}.L5" not in layers or f"{dst_r}.L5" not in layers:
                    continue
                s_ids = layers[f"{src_r}.L5"].ids
                d_ids = layers[f"{dst_r}.L5"].ids
                n = int(min(spec.region_links, len(s_ids), len(d_ids)))
                pick_s = _sample_no_replace(rng, s_ids, n, 1).reshape(-1)
                pick_d = _sample_no_replace(rng, d_ids, n, 1).reshape(-1)
                eng.connect(pick_d, pick_s, rng.uniform(0.2, 0.3, n),
                            group=G.REGION_LINK)

    # ---------------------------------------------------------- 5. 收尾
    eng.finalize()
    pfc_all = np.concatenate([v for v in pfc_groups.values()] + [mpfc])
    enc = Encoder(spec, layout, n_in)
    brain = Brain(engine=eng, spec=spec, geo=geo, ids=ids, layers=layers,
                  pfc_ids=pfc_all, pfc_groups=pfc_groups, motor=motor,
                  encoder=enc)
    if spec.threshold_theta <= 0:
        calibrate_layers(brain)   # 阈值模式下门槛由权重相对阈值定，不再用分位标定
    if spec.pfc_filter:
        # 常亮/常灭的神经元不带"是哪一个输入"的信息，不往前额叶送。
        # 必须在标定之前做：否则门槛会把死单元也算进分位数。
        _filter_invariant(brain)
    k_total = None
    if spec.pfc_topk > 0:
        # 前额叶动态抑制：每拍只留最强的 k 个（字神经元网络的做法）。
        k_total = 0
        for (r, d), g in pfc_groups.items():
            bad = brain.filtered.get(f"{r}.L{d}") if brain.filtered else None
            kept = int((~bad).sum()) if bad is not None else len(g)
            k = max(1, int(round(spec.pfc_topk * kept)))
            eng.register_topk(g, k)
            k_total += k
        eng.rebuild_backend()
    kept_pfc = sum(int((~m).sum()) for m in brain.filtered.values()) if brain.filtered \
        else int(sum(len(v) for v in pfc_groups.values()))
    total_pfc = kept_pfc + nm
    if spec.pfc_gain_div > 0:
        eng.cfg.gain_div = float(spec.pfc_gain_div)
        eng.cfg.gain_div_ids = tuple(int(x) for x in np.concatenate(
            list(pfc_groups.values()) + [mpfc]))
    eng.cfg.target_active = (max(1, int(k_total)) if k_total
                             else max(1, int(round(spec.pfc_target_frac * total_pfc))))
    if do_calibrate:
        cal = calibrate(brain, ticks=calib_ticks, seed=spec.seed + 101)
        if verbose:
            print(f"[ANA] 标定 {calib_ticks} 拍：前额叶门槛={cal['thr_median']:.4f}"
                  f"（中位）  预计活跃≈{cal['pfc_frac']*100:.1f}%")
    if spec.return_adapt > 0 and ids["ret"]:
        # 回程只做"亮一下就过去"的调制：给回程中继自己加习惯化。
        ret_ids = np.concatenate([np.concatenate([v[1], v[2]]) for v in ids["ret"].values()])
        eng.set_adaptation(ret_ids, spec.return_adapt)
        eng._adapt_np[ret_ids] = spec.return_adapt
    if spec.pfc_adapt > 0:
        # 自适应在标定之后开：标定要看原始驱动，不能看"减掉基线"之后的。
        kept_ids = []
        for (r, d), g in pfc_groups.items():
            bad = brain.filtered.get(f"{r}.L{d}") if brain.filtered else None
            kept_ids.append(g if bad is None else g[~bad])
        if kept_ids:
            eng.set_adaptation(np.concatenate(kept_ids), spec.pfc_adapt)
            eng.reset_adaptation()
    if verbose:
        s = brain.summary()
        print(f"[ANA] 神经元 {s['neurons']:,}  边 {s['edges']:,}  "
              f"层 {s['layers']}  前额叶 {s['pfc']:,}  输入细胞 {s['input_cells']:,}")
        print(f"[ANA] 可塑边 {s['plastic_edges']}  前额叶内部边 {s['pfc_internal']}  "
              f"（v0 期望都是 0）")
        for r, L in layers.items():
            print(f"   {r:9s} 每拍选中 {L.k:5d} / {L.size:5d}")
    return brain


# ---------------------------------------------------------------- 参考输入
def _reference_stream(brain: "Brain", n: int, seed: int, dog_only: bool = False) -> list:
    """标定/筛选用的参考输入：一半来自虚拟身体，一半是通用场景。

    只用随机图会把门槛标到"噪声分布"上；只用虚拟身体又会过拟合这个任务。
    """
    from .world import VirtualDog
    sp = brain.spec
    rng = np.random.default_rng(seed)
    dog = VirtualDog(sp, seed=seed)
    out = []
    for i in range(n):
        if dog_only or i % 2 == 0:
            f = (rng.random(sp.n_muscle, dtype=np.float32) * 0.6)
            dog.step(f)
            out.append(dog.observe())
        else:
            out.append(sample_observation(sp, rng, i))
    return out


# ---------------------------------------------------------------- 标定
def _collect_drive(brain: "Brain", ticks: int, rng: np.random.Generator,
                   warmup: int = 18) -> np.ndarray:
    """跑 ticks 拍，记录每个神经元的驱动（不更新门槛，只采样）。"""
    eng, sp, enc = brain.engine, brain.spec, brain.encoder
    buf = np.empty((ticks, eng.n), np.float32)
    stream = _reference_stream(brain, ticks, int(rng.integers(1, 10_000)))
    eng.act[:] = 0.0
    for t, obs in enumerate(stream):
        # 每个场景先"看几拍"让活动沿链传到前额叶（v0 有 5 层 + 两级中继，
        # 传播要 6~8 拍），再采这一拍的驱动。
        for _ in range(warmup):
            eng.clear_external()
            eng.set_input(enc.encode(obs))
            eng.step()
        buf[t] = eng.drive().numpy()
    eng.act[:] = 0.0
    return buf


def calibrate(brain: "Brain", ticks: int = 8, seed: int = 11) -> dict:
    """按"这个神经元自己输入分布的高分位"定门槛——不是写死常数。

    参考实现的经验：门槛必须按神经元自己拿到的信号定，否则会在
    "整层死"和"整层饱和"之间来回跳。这里用驱动分布的 80% 分位当门槛、
    95% 分位当带宽上限，于是每个前额叶神经元只对自己输入里最强的那一档响应
    （组合检测），同时不会因为扇入大小不同而整体偏置。

    门控细胞（I/D）依赖前额叶活动，所以前额叶定完要再采一轮。
    """
    eng = brain.engine
    rng = np.random.default_rng(seed)
    keep_target = eng.cfg.target_active
    eng.cfg.target_active = 0          # 标定时别让整体调控插手中
    buf = _collect_drive(brain, ticks, rng)
    # 1:1 专用线的前额叶 = 直通（门槛已经在 build 里按入边和设好），
    # 不要再用"组内分位数"标定——1:1 的驱动是双峰的（0 或 ~0.9），
    # 分位数会落在峰顶，把整组标成哑的。
    if brain.spec.pfc_fan_in > 1:
        for g in brain.pfc_groups.values():
            _group_thr(eng, g, buf, brain.spec.pfc_target_frac)
    _group_thr(eng, brain.motor["mpfc"], buf, brain.spec.motor_target_frac)

    # 门控细胞依赖前额叶活动，前额叶定完再采一轮
    buf2 = _collect_drive(brain, ticks, rng)
    _group_thr(eng, brain.motor["inh"], buf2, 0.25)
    _group_thr(eng, brain.motor["deinh"], buf2, 0.15)

    # 估一下实际活跃比例（跑 3 拍看）
    eng.act[:] = 0.0
    for t in range(3):
        eng.set_input(brain.encoder.encode(sample_observation(brain.spec, rng, t)))
        eng.step()
    frac = float((eng.act_np[brain.pfc_ids] > 0.35).mean())
    eng.act[:] = 0.0
    eng.gain = 0.0
    eng.cfg.target_active = keep_target
    tmed = float(np.median(eng.thr[brain.pfc_ids]))
    return dict(thr_median=tmed, pfc_frac=frac)


def _group_thr(eng, ids, buf: np.ndarray, frac: float) -> None:
    """按"该组驱动分布的高分位"设一个组门槛：约 frac 比例的神经元过线。"""
    ids = np.asarray(ids, int)
    d = buf[:, ids]
    thr = float(np.quantile(d, 1.0 - float(frac)))
    sd = float(np.std(d))
    # 带宽：太窄会变成二值抖动，太宽会整组同时亮
    soft = max(0.5 * sd, 0.05 * max(thr, 1e-6), 1e-3)
    eng.set_thr(ids, thr=np.full(len(ids), thr, np.float32),
                soft=np.full(len(ids), soft, np.float32))


# ---------------------------------------------------------------- 常亮筛选
def _filter_invariant(brain: "Brain", ticks: int | None = None) -> dict:
    """常亮/常灭的神经元不往前额叶送（按跨输入 std 筛，不按当前亮不亮）。

    依据：输入层这类"永远一半亮"的细胞，跨输入几乎没有起伏，不带"是哪一个
    输入"的信息；把它们送进前额叶只会让联想连接被背景淹没。

    只有 1:1 模式能做：中继和前额叶跟层神经元同序号，按序号剪就行。
    组合模式（fan-in>1）不做筛选，留待以后按 PFC 单元统计。
    """
    sp, eng = brain.spec, brain.engine
    if not (sp.pfc_fan_in == 1 and abs(sp.pfc_ratio - 1.0) < 1e-9):
        return {}
    ticks = int(ticks or sp.filter_ticks)
    rates = {name: np.zeros(L.size, np.float32) for name, L in brain.layers.items()}
    # 参考分布要"杂"：只用身体截图会让绝大部分神经元永远不亮，
    # 它们会被误判成"常灭"而剪掉（实测只剩 2%）。一半身体、一半通用场景。
    stream = _reference_stream(brain, ticks, sp.seed + 77, dog_only=False)
    eng.act[:] = 0.0
    for obs in stream:
        for _ in range(18):         # 让活动传到最后一层（5 层 + 两级中继，深链要 15 拍左右）
            eng.set_input(brain.encoder.encode(obs))
            eng.step()
        a = eng.act_np
        for name, L in brain.layers.items():
            rates[name] += (a[L.ids] > 0.35)
    eng.act[:] = 0.0

    drop_src, drop_dst = [], []
    filtered: dict[str, np.ndarray] = {}
    for name, L in brain.layers.items():
        p = rates[name] / ticks
        # 常灭（几乎不出现在任何经历里）和常亮（几乎总在）都不送前额叶：
        # 前者是死单元，后者不带"是哪一个输入"的信息。
        bad = (p < sp.filter_min_rate) | (p > sp.filter_max_rate)
        filtered[name] = bad
        if not bad.any():
            continue
        idx = np.flatnonzero(bad)
        r1, r2 = brain.ids["relay"][(L.region, L.depth)]
        pfc = brain.pfc_groups[(L.region, L.depth)]
        if len(r1) != L.size or len(pfc) != L.size:
            filtered[name] = np.zeros(L.size, bool)
            continue
        drop_src += [L.ids[idx], r1[idx], r2[idx]]
        drop_dst += [r1[idx], r2[idx], pfc[idx]]
        eng.set_thr(pfc[idx], thr=1e9, soft=1.0)   # 死单元：永远过不了门槛
        eng._counted_f[pfc[idx]] = False
    if drop_src:
        n = eng.n
        src = np.concatenate(drop_src).astype(np.int64)
        dst = np.concatenate(drop_dst).astype(np.int64)
        keys = src * n + dst
        ek = eng.src.astype(np.int64) * n + eng.dst.astype(np.int64)
        eng.drop_edges(np.isin(ek, keys))
        eng.rebuild_backend()
    brain.filtered = filtered
    return filtered


def calibrate_layers(brain: "Brain", ticks: int = 8) -> None:
    """层内门槛/带宽按"这个神经元自己实测的驱动"定。

    链式架构的原话：门槛/带宽要按该神经元自己拿到的信号定，不能用整层统计量。
    做法：跑几个参考场景，量出每个神经元驱动的时间平均，把带宽设成这个平均值，
    于是"典型驱动"恰好对应满响应；再由 top-k 决定谁被选中。
    不这样做，驱动会沿"5 层 + 两级中继"逐层衰减（实测每层掉一半）。
    """
    eng = brain.engine
    rng = np.random.default_rng(brain.spec.seed + 31)
    buf = _collect_drive(brain, ticks, rng, warmup=18)
    # 带宽 = 这个神经元自己驱动的 75% 分位。
    # 注：链式库 calib.py 的公式是"门槛=0.7×中位、带宽=2×起伏+0.15×中位"。
    # 照抄试过（2026-10-22）：活动会落在分级区，同一张图连看三次亮的单元会差几个
    # （1198/1212/1212）——浮点抖动改变了 top-k 的选中结果，破坏"同一输入→同一批
    # 神经元"。所以保留"接近饱和"的带宽，偏差记在 docs/AUDIT.md。
    q = np.quantile(buf, 0.75, axis=0)
    for name, L in brain.layers.items():
        m = np.maximum(q[L.ids], 1e-4)
        eng.set_thr(L.ids, thr=0.0, soft=m.astype(np.float32))