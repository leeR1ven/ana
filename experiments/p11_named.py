# -*- coding: utf-8 -*-
"""P11：概念名表——标定、起名、存盘、按名装规则、换个进程也能用。

  1. 看画面 A → 起名"场景A"，看画面 B → 起名"场景B"；
  2. 存成 JSON；
  3. 用名字写规则：场景A → 站（5 档），场景B → 撑（7 档）；
  4. 装进网络，点 A 得 0.50、点 B 得 0.70；
  5. 新开一个脑子（同一个种子），只读名字文件重新装一遍 —— 结果一样。
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec                                   # noqa: E402
from agi.naming import NameTable                                    # noqa: E402
from experiments.p1_recall import scene_obs                          # noqa: E402

TMP = os.path.join(ROOT, "_names.json")


def forces_for(eng, brain, ids, ticks=12):
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(np.zeros(brain.encoder.n_in, np.float32))
        eng.set_external(ids, np.ones(len(ids), np.float32))
        eng.step()
    return brain.forces().copy()


def install_all(eng, brain, nt: NameTable) -> int:
    rules = []
    for name, levels in (("场景A", 5), ("场景B", 7)):
        for m in range(brain.spec.n_muscle):
            rules.append((f"{name}#{m}", brain.motor["force"][m][:levels], 6.0))
    # 每个名字只连自己的模块（这里简化：名称已带肌肉号，实际用时按名字取单元集合）
    total = 0
    others = {"场景A": "场景B", "场景B": "场景A"}
    for name, levels in (("场景A", 5), ("场景B", 7)):
        src = np.array(sorted(set(nt.get(name).tolist()) - set(nt.get(others[name]).tolist())), int)
        for m in range(brain.spec.n_muscle):
            dst = brain.motor["force"][m][:levels]
            eng_src = np.repeat(src, len(dst))
            eng_dst = np.tile(dst, len(src))
            eng.add_edges(eng_src.astype(np.int32), eng_dst.astype(np.int32),
                          np.full(len(eng_src), 6.0, np.float32), group=61)
            total += len(eng_src)
    if total:
        eng.rebuild_backend()
    return total


def main() -> int:
    sp = build_spec("full")
    brain = build(sp, verbose=False)
    nt = NameTable(TMP)
    nt.calibrate(brain, scene_obs(sp, 1), "场景A")
    nt.calibrate(brain, scene_obs(sp, 2), "场景B")
    nt.save(TMP)
    print(f"标定并保存：{[(k, len(v)) for k, v in nt.names.items()]} → {TMP}")

    n = install_all(brain.engine, brain, nt)
    fA = forces_for(brain.engine, brain, nt.get("场景A"))
    fB = forces_for(brain.engine, brain, nt.get("场景B"))
    print(f"装 {n} 条边：点 A → {fA.mean():.2f}（期望 0.50）；点 B → {fB.mean():.2f}（期望 0.70）")

    # 换个脑子，只读名字文件
    brain2 = build(sp, verbose=False)
    nt2 = NameTable.load(TMP)
    n2 = install_all(brain2.engine, brain2, nt2)
    gA = forces_for(brain2.engine, brain2, nt2.get("场景A"))
    gB = forces_for(brain2.engine, brain2, nt2.get("场景B"))
    print(f"新脑子重装 {n2} 条边：点 A → {gA.mean():.2f}；点 B → {gB.mean():.2f}")
    ok = abs(gA.mean() - fA.mean()) < 0.02 and abs(gB.mean() - fB.mean()) < 0.02
    print(f"结论：{'名字表可用（标定→存盘→换脑子重装，结果一致）' if ok else '重装结果不一致'}")
    os.remove(TMP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
