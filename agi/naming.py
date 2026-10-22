# -*- coding: utf-8 -*-
"""概念名表：把"哪些神经元代表某个东西"存成名字，之后用名字写规则、装进网络。

为什么需要（born-wired-cortex 的教训）：皮层是重新编码的，**不能按输入编号写本能**
（实测满屏红时，亮在"输入层红色那一段"的比例和乱猜差不多）。正确做法是：

    1. 标定：给它看一个例子，记下哪些神经元亮了，起个名字；
    2. 用名字写规则：  视觉:红 -> 运动:站姿  1.0  固定
    3. 装：把规则展开成一根根连接。

这里只做"名字"这一层（存文件、按名取单元、按规则装边），规则表是最简的元组列表。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NameTable:
    """名字 → 神经元编号集合。可以存成 JSON、换个进程再读回来。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.names: dict[str, list[int]] = {}

    # ---------------------------------------------------------- 标定
    def calibrate(self, brain, obs, name: str) -> np.ndarray:
        """让脑子看一个例子，把它亮的前额叶单元记在这个名字下。"""
        from agi.loop import Cortex          # 延迟导入，避免循环

        ctx = Cortex(brain)
        for _ in range(12):          # 码要几拍才稳定
            ctx.tick(obs, record=False)
        ids = np.asarray(brain.pfc_code(), np.int64)
        self.names[name] = [int(x) for x in ids]
        return ids

    def get(self, name: str) -> np.ndarray:
        if name not in self.names:
            raise KeyError(f"没有这个名字：{name}")
        return np.asarray(self.names[name], np.int64)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return len(self.names)

    # ---------------------------------------------------------- 存/取
    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else self.path
        if p is None:
            raise ValueError("没有给路径")
        p.write_text(json.dumps(self.names, ensure_ascii=False, indent=1), encoding="utf-8")
        self.path = p
        return p

    @classmethod
    def load(cls, path: str | Path) -> "NameTable":
        t = cls(path)
        t.names = {k: [int(x) for x in v]
                   for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}
        return t

    # ---------------------------------------------------------- 装
    def install(self, engine, rules, group: int = 61, verbose: bool = False) -> int:
        """按名字把规则装成连接。

        rules: 可迭代的 (源名字, 目标神经元编号数组, 权重) ——
               目标通常是一块肌肉的力度档，用 `brain.motor["force"][m][:k]` 取。
        """
        total = 0
        for src_name, dst_ids, w in rules:
            src = self.get(src_name)
            dst_ids = np.asarray(dst_ids, np.int64)
            eng_src = np.repeat(src, len(dst_ids))
            eng_dst = np.tile(dst_ids, len(src))
            engine.add_edges(eng_src.astype(np.int32), eng_dst.astype(np.int32),
                             np.full(len(eng_src), float(w), np.float32), group=group)
            total += len(eng_src)
            if verbose:
                print(f"  装 {src_name}: {len(src)} 单元 × {len(dst_ids)} 目标 = "
                      f"{len(eng_src)} 条边（权重 {w}）")
        if total:
            engine.rebuild_backend()
        return total


class RuleTable:
    """规则表：一行一条"名字 → 动作"，可存文件、可换进程重装。

    一行的格式：{"src": "站", "muscles": "all" 或 [0,1,...], "levels": 5, "weight": 6.0}
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.rules: list[dict] = []

    def add(self, src: str, levels: int, weight: float = 6.0,
            muscles="all") -> None:
        self.rules.append(dict(src=src, muscles=muscles, levels=int(levels),
                               weight=float(weight)))

    def save(self, path=None) -> Path:
        p = Path(path) if path else self.path
        if p is None:
            raise ValueError("没有给路径")
        p.write_text(json.dumps(self.rules, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        self.path = p
        return p

    @classmethod
    def load(cls, path) -> "RuleTable":
        t = cls(path)
        t.rules = json.loads(Path(path).read_text(encoding="utf-8"))
        return t

    def install(self, engine, brain, table: "NameTable", group: int = 62,
                max_sources: int = 660) -> int:
        """按规则把"名字"展开成"名字里的单元 → 各肌肉的力度档"。

        源单元数要**封顶**：阈值模式下前额的码可能大好几倍，若按"每个源都连一条
        固定权重的边"，力度细胞会被推满（实测点"站"从 0.5 变成 1.0）。
        这里均匀抽稀到 max_sources，保证与标定时的手感一致。
        """
        total = 0
        for r in self.rules:
            src = table.get(r["src"])
            if len(src) > max_sources:
                keep = np.linspace(0, len(src) - 1, max_sources).astype(int)
                src = src[keep]
            muscles = (range(brain.spec.n_muscle) if r["muscles"] == "all"
                       else r["muscles"])
            for m in muscles:
                dst = brain.motor["force"][m][: int(r["levels"])]
                s = np.repeat(src, len(dst))
                d = np.tile(dst, len(src))
                # 权重按源数量均分：规则里的 weight 表示"总驱动"，与源多少无关
                w = float(r["weight"]) / max(1, len(src))
                engine.add_edges(s.astype(np.int32), d.astype(np.int32),
                                 np.full(len(s), w, np.float32), group=group)
                total += len(s)
        if total:
            engine.rebuild_backend()
        return total

    def __len__(self) -> int:
        return len(self.rules)