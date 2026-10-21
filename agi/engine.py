# -*- coding: utf-8 -*-
"""引擎：神经元 + 有向边 + 一拍动力学。

引擎只管"有什么、怎么走"，不管"为什么这么连"（那是 build.py 的事）。

几条从参考实现里学来的硬规则（都有实测依据，见 docs/MECHANISMS.md）：

1. **权重随机、每条边各不相同、不做归一化**。归一化会让"此消彼长"的两路
   信号加起来变成常数，画面差别就被加没了。
2. **就近只看 (y,z) 平面**，层与层之间固定的 x 差不参与距离——否则每个
   神经元看到的都是上一层最靠边的一排，图像的"行"就丢了。
3. **门槛/带宽从连线本身算**（该神经元入边权重和 × 比例），不用整层统计量
   去标定——那不是在饱和就是在沉底。
4. **抑制是独立的抑制细胞**，不在兴奋连接上写负权重；抑制细胞的输出取负。
5. **每层精确 top-k**：每拍被选中的神经元个数恒定（用户要求：不允许有的
   图点亮很多、有的图点亮很少，否则等效于梯度爆炸/消失）。
6. **稳定 ID**：删除只改 alive、不移位；新增拿新编号。消费方先按 alive 过滤。

后端：默认 torch（CPU，index_add_ 比 numpy 快约 3 倍）；没有 torch 时退回
numpy（`np.bincount`）。两条路径的数值语义一致。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - 取决于环境
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


KIND = {
    "input": 0,
    "l1": 1,
    "layer": 2,
    "relay": 3,
    "pfc": 4,
    "ret_inh": 5,
    "ret_relay": 6,
    "motor_relay": 7,
    "force": 8,
    "motor_inh": 9,
    "motor_deinh": 10,
    "clock": 11,
}
KIND_NAME = {v: k for k, v in KIND.items()}


@dataclass
class EngineConfig:
    tau: float = 0.03             # 活动时间常数
    dt: float = 0.02              # 一拍 = 50 Hz
    gain_lr: float = 0.002        # 整体阈值调控速度
    gain_range: tuple = (-0.05, 0.05)   # 只许微调（±0.5 会把深层压死）
    target_active: int = 0        # 0 = 不调控
    backend: str = "auto"         # auto | torch | numpy
    seed: int = 7


class Engine:
    """神经元与边。构建期用列表追加，`finalize()` 后进入可跑状态。"""

    def __init__(self, cfg: EngineConfig | None = None):
        self.cfg = cfg or EngineConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._n = 0
        self._region_names: list[str] = []
        self._kind: list[np.ndarray] = []
        self._region: list[np.ndarray] = []
        self._depth: list[np.ndarray] = []
        self._thr: list[np.ndarray] = []
        self._soft: list[np.ndarray] = []
        self._inh: list[np.ndarray] = []
        self._counted: list[np.ndarray] = []
        self._pos: list[np.ndarray] = []
        self._e_src: list[np.ndarray] = []
        self._e_dst: list[np.ndarray] = []
        self._e_w: list[np.ndarray] = []
        self._e_sgn: list[np.ndarray] = []
        self._e_group: list[np.ndarray] = []
        self._topk: list[tuple[int, int, int]] = []
        self._final = False
        self.gain = 0.0
        self._backend = "numpy"
        self.input_size = 0          # 输入细胞是编号最前面的一段，每拍直接赋值（不滞后）
        self._gain_tick = 0
        # 增量缓存（构建期省掉重复拼接）
        self._pos_flat = np.zeros((0, 3), np.float32)
        self._inh_flat = np.zeros(0, bool)
        self._in_sum = np.zeros(0, np.float32)
        # 运行期（finalize 后）
        self.act = np.zeros(0, np.float32)
        self.ext = np.zeros(0, np.float32)
        self.selected = np.zeros(0, bool)     # 本拍被 top-k 选中的神经元
        # 自适应（习惯化）：驱动减掉自己的滑动平均，再比门槛/top-k
        self.adapt_rate = 0.0
        self._base = np.zeros(0, np.float32)
        self._adapt_np = np.zeros(0, np.float32)
        self._base_t = None
        self._adapt_t = None
        self._act_t = self._ext_t = self._tbuf = None
        self._src_t = self._dst_t = self._wp_t = self._sgn_t = None
        self._thr_t = self._soft_t = self._counted_t = None
        self._masks: list[tuple[int, int, np.ndarray, object]] = []

    # ------------------------------------------------------------ 构建
    @property
    def n(self) -> int:
        return self._n

    @property
    def region_names(self) -> list[str]:
        return list(self._region_names)

    def region_id(self, name: str) -> int:
        try:
            return self._region_names.index(name)
        except ValueError:
            self._region_names.append(name)
            return len(self._region_names) - 1

    def add_block(self, size: int, *, kind: str, region: str = "", depth: int = 0,
                  thr: float = 0.30, soft: float = 0.30, inh: bool = False,
                  counted: bool = True, pos=None) -> np.ndarray:
        if self._final:
            raise RuntimeError("已 finalize，不能再加神经元（结构可塑性走 grow_units）")
        size = int(size)
        if size <= 0:
            return np.zeros(0, np.int32)
        ids = np.arange(self._n, self._n + size, dtype=np.int32)
        rid = self.region_id(region) if region else -1
        self._kind.append(np.full(size, KIND[kind], np.int8))
        self._region.append(np.full(size, rid, np.int16))
        self._depth.append(np.full(size, depth, np.int8))
        self._thr.append(np.full(size, float(thr), np.float32))
        self._soft.append(np.full(size, float(soft), np.float32))
        self._inh.append(np.full(size, bool(inh)))
        self._counted.append(np.full(size, bool(counted)))
        p = np.zeros((size, 3), np.float32) if pos is None else np.asarray(pos, np.float32)
        if p.shape != (size, 3):
            raise ValueError(f"pos 形状 {p.shape} 与 size {size} 不符")
        self._pos.append(p)
        self._pos_flat = np.vstack([self._pos_flat, p]) if len(self._pos_flat) else p.copy()
        self._inh_flat = np.r_[self._inh_flat, np.full(size, bool(inh))]
        self._in_sum = np.r_[self._in_sum, np.zeros(size, np.float32)]
        self._n += size
        return ids

    def connect(self, dst, src, w, group: int = 0) -> int:
        """显式连边：dst/src/w 一一对应。返回加的边数。"""
        dst = np.asarray(dst, np.int32).ravel()
        src = np.asarray(src, np.int32).ravel()
        w = np.asarray(w, np.float32).ravel()
        if not (len(dst) == len(src) == len(w)):
            raise ValueError(f"connect 长度不一致：{len(dst)}/{len(src)}/{len(w)}")
        if len(dst) == 0:
            return 0
        g = (np.full(len(dst), int(group), np.int16) if np.isscalar(group)
             else np.asarray(group, np.int16).ravel())
        sgn = np.where(self.is_inh[src], -1.0, 1.0).astype(np.float32)
        self._e_src.append(src)
        self._e_dst.append(dst)
        self._e_w.append(w)
        self._e_sgn.append(sgn)
        self._e_group.append(g)
        self._in_sum += np.bincount(dst, weights=np.abs(w),
                                    minlength=self._n).astype(np.float32)
        return len(dst)

    def link(self, src, dst, w=None, group: int = 0) -> int:
        """一对一专用线：第 i 个 source 只连第 i 个 target。"""
        src = np.asarray(src, np.int32).ravel()
        dst = np.asarray(dst, np.int32).ravel()
        if len(src) != len(dst):
            raise ValueError(f"link 要求一一对应：{len(src)} vs {len(dst)}")
        ww = (np.full(len(src), 1.0, np.float32) if w is None
              else np.asarray(w, np.float32).ravel())
        return self.connect(dst, src, ww, group)

    def add_edges(self, src, dst, w, group: int = 0) -> int:
        """运行期加边（可塑连接用）。加完要 rebuild_backend()。"""
        src = np.asarray(src, np.int32).ravel()
        dst = np.asarray(dst, np.int32).ravel()
        w = np.asarray(w, np.float32).ravel()
        if not (len(src) == len(dst) == len(w)):
            raise ValueError("add_edges 长度不一致")
        if not len(src):
            return 0
        sgn = np.where(self.is_inh[src], -1.0, 1.0).astype(np.float32)
        self.src = np.r_[self.src, src]
        self.dst = np.r_[self.dst, dst]
        self.w = np.r_[self.w, w]
        self.wp = np.r_[self.wp, w]
        self.sgn = np.r_[self.sgn, sgn]
        self.group = np.r_[self.group, np.full(len(src), int(group), np.int16)]
        if len(self._in_sum) == self._n:
            self._in_sum = self._in_sum + np.bincount(
                dst, weights=np.abs(w), minlength=self._n).astype(np.float32)
        return len(src)

    def drop_edges(self, mask) -> int:
        """按掩码删边（True = 删）。删完要 rebuild_backend()。"""
        mask = np.asarray(mask, bool)
        keep = ~mask
        n = int(mask.sum())
        if n == 0:
            return 0
        self.src = self.src[keep]
        self.dst = self.dst[keep]
        self.w = self.w[keep]
        self.wp = self.wp[keep]
        self.sgn = self.sgn[keep]
        self.group = self.group[keep]
        return n

    def set_adaptation(self, ids, rate: float) -> None:
        """给这些神经元开自适应（每拍减掉自己的滑动平均驱动）。"""
        ids = np.asarray(ids, int)
        self.adapt_rate = float(rate)
        self._adapt_np[ids] = float(rate)
        if self._adapt_t is not None:
            self._adapt_t.copy_(torch.from_numpy(self._adapt_np))

    def reset_adaptation(self) -> None:
        self._base[:] = 0.0
        if self._base_t is not None:
            self._base_t.zero_()

    def register_topk(self, ids, k: int) -> None:
        ids = np.asarray(ids, np.int64)
        if len(ids) == 0:
            return
        if ids[-1] - ids[0] + 1 != len(ids):
            raise ValueError("top-k 层必须是连续编号块")
        k = int(k)
        if k < len(ids):
            self._topk.append((int(ids[0]), int(ids[-1]) + 1, k))

    # ------------------------------------------------------------ 构建期可读
    def _cat(self, chunks: list[np.ndarray], empty, dtype) -> np.ndarray:
        return np.concatenate(chunks).astype(dtype, copy=False) if chunks else empty

    @property
    def kind(self) -> np.ndarray:
        return self._kind_f if self._final else self._cat(self._kind, np.zeros(0, np.int8), np.int8)

    @property
    def is_inh(self) -> np.ndarray:
        return self._inh_f if self._final else self._inh_flat

    @property
    def pos(self) -> np.ndarray:
        return self._pos_f if self._final else self._pos_flat

    @property
    def region(self) -> np.ndarray:
        return self._region_f if self._final else self._cat(self._region, np.zeros(0, np.int16), np.int16)

    @property
    def depth(self) -> np.ndarray:
        return self._depth_f if self._final else self._cat(self._depth, np.zeros(0, np.int8), np.int8)

    def in_weight_sum(self, ids=None) -> np.ndarray:
        """每个神经元的入边权重和（用绝对值；抑制边也算"接了多少线"）。"""
        if ids is None:
            return self._in_sum
        return self._in_sum[np.asarray(ids, int)]

    def set_thr(self, ids, thr=None, soft=None, thr_frac=None, soft_frac=None) -> None:
        """设置门槛/带宽。可以给标量、按神经元的数组，或"入边权重和 × 比例"。

        finalize 之后也能改（会同步到后端张量）——标定阶段要用。
        """
        ids = np.asarray(ids, np.int64)
        if len(ids) == 0:
            return
        if thr_frac is not None or soft_frac is not None:
            s = self.in_weight_sum(ids)
            if thr_frac is not None:
                self._put_thr(ids, (thr_frac * s).astype(np.float32))
            if soft_frac is not None:
                self._put_soft(ids, np.maximum(soft_frac * s, 1e-4).astype(np.float32))
            return
        if thr is not None:
            self._put_thr(ids, self._as_values(thr, len(ids)))
        if soft is not None:
            self._put_soft(ids, self._as_values(soft, len(ids)))

    def _put_thr(self, ids: np.ndarray, vals: np.ndarray) -> None:
        if self._final:
            self._thr_f[ids] = vals
            if self._thr_t is not None:
                self._thr_t[torch.as_tensor(ids)] = torch.as_tensor(vals)
        else:
            self._write(self._thr, ids, vals.astype(np.float32))

    def _put_soft(self, ids: np.ndarray, vals: np.ndarray) -> None:
        if self._final:
            self._soft_f[ids] = vals
            if self._soft_t is not None:
                self._soft_t[torch.as_tensor(ids)] = torch.as_tensor(vals)
        else:
            self._write(self._soft, ids, vals.astype(np.float32))

    @staticmethod
    def _as_values(v, n: int) -> np.ndarray:
        a = np.asarray(v, np.float32).ravel()
        if a.size == 1:
            return np.full(n, float(a[0]), np.float32)
        if a.size != n:
            raise ValueError(f"期望 1 个或 {n} 个值，拿到 {a.size}")
        return a.astype(np.float32)

    @staticmethod
    def _write(chunks: list[np.ndarray], ids: np.ndarray, values: np.ndarray) -> None:
        """把全局编号 values 写回分块数组（构建期，块数很少，直接整体重排）。"""
        arr = np.concatenate(chunks)
        arr[ids] = values
        pos = 0
        for i, c in enumerate(chunks):
            chunks[i] = arr[pos:pos + len(c)]
            pos += len(c)

    # ------------------------------------------------------------ 进入运行期
    def finalize(self) -> "Engine":
        if self._final:
            return self
        self._kind_f = self._cat(self._kind, np.zeros(0, np.int8), np.int8)
        self._region_f = self._cat(self._region, np.zeros(0, np.int16), np.int16)
        self._depth_f = self._cat(self._depth, np.zeros(0, np.int8), np.int8)
        self._thr_f = self._cat(self._thr, np.zeros(0, np.float32), np.float32)
        self._soft_f = self._cat(self._soft, np.zeros(0, np.float32), np.float32)
        self._inh_f = self._cat(self._inh, np.zeros(0, bool), bool)
        self._counted_f = self._cat(self._counted, np.zeros(0, bool), bool)
        self._pos_f = np.vstack(self._pos) if self._pos else np.zeros((0, 3), np.float32)
        self._alive_f = np.ones(self._n, bool)
        self.src = self._cat(self._e_src, np.zeros(0, np.int32), np.int32)
        self.dst = self._cat(self._e_dst, np.zeros(0, np.int32), np.int32)
        self.w = self._cat(self._e_w, np.zeros(0, np.float32), np.float32)
        self.wp = self.w.copy()
        self.sgn = self._cat(self._e_sgn, np.zeros(0, np.float32), np.float32)
        self.group = self._cat(self._e_group, np.zeros(0, np.int16), np.int16)
        # 去重：同一对 (src,dst) 只保留一条（多次采样撞上时留第一条）
        if len(self.src):
            key = self.src.astype(np.int64) * max(1, self._n) + self.dst
            _, first = np.unique(key, return_index=True)
            if len(first) < len(key):
                keep = np.zeros(len(key), bool)
                keep[first] = True
                self.src = self.src[keep]
                self.dst = self.dst[keep]
                self.w = self.w[keep]
                self.wp = self.wp[keep]
                self.sgn = self.sgn[keep]
                self.group = self.group[keep]
        self.act = np.zeros(self._n, np.float32)
        self.ext = np.zeros(self._n, np.float32)
        self.selected = np.zeros(self._n, bool)
        self._base = np.zeros(self._n, np.float32)
        self._adapt_np = np.zeros(self._n, np.float32)
        self._final = True
        self._build_backend()
        return self

    def _build_backend(self) -> None:
        want = self.cfg.backend
        self._backend = "torch" if (want in ("auto", "torch") and _HAS_TORCH) else "numpy"
        self._masks = []
        if self._backend == "torch":
            self._act_t = torch.from_numpy(self.act)
            self._ext_t = torch.from_numpy(self.ext)
            self._src_t = torch.as_tensor(self.src.astype(np.int64))
            self._dst_t = torch.as_tensor(self.dst.astype(np.int64))
            self._wp_t = torch.as_tensor(self.wp)
            self._sgn_t = torch.as_tensor(self.sgn)
            self._thr_t = torch.as_tensor(self._thr_f)
            self._soft_t = torch.as_tensor(self._soft_f)
            self._counted_t = torch.from_numpy(self._counted_f)
            self._base_t = torch.from_numpy(self._base)
            self._adapt_t = torch.from_numpy(self._adapt_np)
            self._tbuf = torch.zeros(self._n, dtype=torch.float32)
            for s, e, k in self._topk:
                m = np.zeros(e - s, np.float32)
                self._masks.append((s, e, k, torch.from_numpy(m)))
        else:
            self._masks = []
            for s, e, k in self._topk:
                self._masks.append((s, e, k, np.zeros(e - s, np.float32)))

    # ------------------------------------------------------------ 一拍
    def drive(self):
        """当前驱动（入边贡献 + 外部）。step 与标定都用它，保证同一套数值。"""
        if not self._final:
            raise RuntimeError("先 finalize()")
        if self._backend == "torch":
            b = self._tbuf
            b.zero_()
            b.index_add_(0, self._dst_t, self._wp_t * self._sgn_t * self._act_t[self._src_t])
            b.add_(self._ext_t)
            if self.adapt_rate > 0.0:
                b.sub_(self._base_t)                 # 只保留"相对自己的变化"
                self._base_t.add_(b * self._adapt_t)  # base += rate·(raw-base)
            return b
        d = np.bincount(self.dst, weights=self.wp * self.sgn * self.act[self.src],
                        minlength=self._n).astype(np.float32)
        d += self.ext
        if self.adapt_rate > 0.0:
            raw = d.copy()
            d -= self._base
            self._base += self._adapt_np * (raw - self._base)
        return d

    def step(self) -> None:
        c = self.cfg
        if not self._final:
            raise RuntimeError("先 finalize()")
        if self._backend == "torch":
            b = self.drive()
            eff = torch.clamp(self._thr_t * (1.0 + self.gain), min=1e-4)
            raw = torch.clamp((b - eff) / self._soft_t, 0.0, 1.0)
            rn = raw.numpy()
            self.selected[:] = False
            for s, e, k, mt in self._masks:
                seg = rn[s:e]
                idx = np.argpartition(-seg, k - 1)[:k]
                mt.zero_()
                mt[idx] = 1.0
                raw[s:e].mul_(mt)
                self.selected[s:e] = mt.numpy().astype(bool)
            # act += (dt/tau)(raw-act)，原地做，不额外分配
            self._act_t.lerp_(raw, c.dt / c.tau)
            self._act_t.clamp_(0.0, 1.0)
        else:
            drive = self.drive()
            eff = np.maximum(self._thr_f * (1.0 + self.gain), 1e-4)
            raw = np.clip((drive - eff) / self._soft_f, 0.0, 1.0).astype(np.float32)
            self.selected[:] = False
            for s, e, k, m in self._masks:
                seg = raw[s:e]
                idx = np.argpartition(-seg, k - 1)[:k]
                m[:] = 0.0
                m[idx] = 1.0
                seg *= m
                self.selected[s:e] = m > 0
            self.act += (c.dt / c.tau) * (raw - self.act)
            np.clip(self.act, 0.0, 1.0, out=self.act)
        if self.input_size:
            # 感觉细胞是"当前读数"，不是积分量：每拍直接等于输入，不滞后。
            n = self.input_size
            if self._backend == "torch":
                self._act_t[:n].copy_(self._ext_t[:n])
            else:
                self.act[:n] = self.ext[:n]
        self._gain_tick += 1
        if c.target_active and (self._gain_tick % 4 == 0):
            self._gain_step()

    def _gain_step(self) -> None:
        c = self.cfg
        if self._backend == "torch":
            a = self._act_t
            n_on = int(((a > 0.35) & self._counted_t).sum())
            n_sat = int(((a > 0.98) & self._counted_t).sum())
        else:
            a = self.act
            n_on = int(((a > 0.35) & self._counted_f).sum())
            n_sat = int(((a > 0.98) & self._counted_f).sum())
        tgt = max(1, c.target_active)
        self.gain += c.gain_lr * (n_on / tgt - 1.0)
        self.gain += c.gain_lr * 2.0 * (n_sat / tgt - 0.1)
        self.gain = float(np.clip(self.gain, *c.gain_range))

    # ------------------------------------------------------------ 读
    @property
    def act_np(self) -> np.ndarray:
        return self._act_t.numpy() if self._backend == "torch" else self.act

    @property
    def alive(self) -> np.ndarray:
        return self._alive_f

    @property
    def thr(self) -> np.ndarray:
        return self._thr_f

    @property
    def soft(self) -> np.ndarray:
        return self._soft_f

    @property
    def n_edges(self) -> int:
        return int(len(self.src))

    def set_input(self, values) -> None:
        """写入感觉输入（引擎最前面 input_size 个细胞）。"""
        n = self.input_size
        v = np.asarray(values, np.float32).ravel()
        if len(v) != n:
            raise ValueError(f"set_input 期望 {n} 个值，拿到 {len(v)}")
        self.ext[:n] = v
        if self._backend == "torch":
            self._ext_t[:n].copy_(torch.as_tensor(v))

    def set_external(self, ids, values) -> None:
        """给非感觉细胞加外部驱动（例如反射直接压到运动区）。"""
        ids = np.asarray(ids, np.int64)
        v = np.asarray(values, np.float32)
        if len(ids) == 0:
            return
        self.ext[ids] = v
        if self._backend == "torch":
            self._ext_t[torch.as_tensor(ids)] = torch.as_tensor(v)
    def clear_external(self) -> None:
        self.ext[:] = 0.0
        if self._backend == "torch":
            self._ext_t.zero_()

    def active(self, ids=None, thr: float = 0.35) -> np.ndarray:
        a = self.act_np
        if ids is None:
            return np.flatnonzero(a > thr)
        ids = np.asarray(ids, int)
        return ids[a[ids] > thr]

    def count_active(self, ids, thr: float = 0.35) -> int:
        return int((self.act_np[np.asarray(ids, int)] > thr).sum())

    # ------------------------------------------------------------ 结构可塑性（以后用）
    def grow_units(self, n_new: int, *, kind: str, region: str, depth: int,
                   thr: float, soft: float, pos, inh: bool = False,
                   counted: bool = True) -> np.ndarray:
        """运行期加神经元：稳定 ID（只追加，不移位）。加完边要 `rebuild_backend()`。"""
        if not self._final:
            raise RuntimeError("构建期直接 add_block")
        size = int(n_new)
        ids = np.arange(self._n, self._n + size, np.int32)
        self._kind_f = np.r_[self._kind_f, np.full(size, KIND[kind], np.int8)]
        self._region_f = np.r_[self._region_f, np.full(size, self.region_id(region), np.int16)]
        self._depth_f = np.r_[self._depth_f, np.full(size, depth, np.int8)]
        self._thr_f = np.r_[self._thr_f, np.full(size, thr, np.float32)]
        self._soft_f = np.r_[self._soft_f, np.full(size, soft, np.float32)]
        self._inh_f = np.r_[self._inh_f, np.full(size, inh)]
        self._counted_f = np.r_[self._counted_f, np.full(size, counted)]
        self._pos_f = np.vstack([self._pos_f, np.asarray(pos, np.float32)])
        self._alive_f = np.r_[self._alive_f, np.ones(size, bool)]
        self.act = np.r_[self.act_np, np.zeros(size, np.float32)]
        self.ext = np.r_[self.ext, np.zeros(size, np.float32)]
        self._n += size
        self.rebuild_backend()
        return ids

    def rebuild_backend(self) -> None:
        """边或神经元变了以后重建后端张量（活动值保留）。"""
        keep = self.act_np.copy()
        self._build_backend()
        self.act[:] = keep
        if self._backend == "torch":
            self._act_t.copy_(torch.from_numpy(self.act))

    def stats(self) -> dict:
        return dict(neurons=self._n, edges=self.n_edges, backend=self._backend,
                    gain=float(self.gain), active=int((self.act_np > 0.35).sum()))
