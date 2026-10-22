# -*- coding: utf-8 -*-
"""P14：规则表——"名字 → 动作"写成文件，换个进程重新装。

  1. 标定"站"，存名字表；
  2. 写规则表：站 → 12 块肌肉 × 5 档，权重 6；
  3. 存 JSON；
  4. 新开脑子 + 只读两个 JSON，重装 → 点"站"得 0.5 左右，狗站得住。
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import build, build_spec                                     # noqa: E402
from agi.naming import NameTable, RuleTable                           # noqa: E402
from experiments.p1_recall import scene_obs                           # noqa: E402

NP = os.path.join(ROOT, "_names_r.json")
RP = os.path.join(ROOT, "_rules_r.json")


def main() -> int:
    sp = build_spec("full")
    b1 = build(sp, verbose=False)
    nt = NameTable(NP)
    nt.calibrate(b1, scene_obs(sp, 1), "站")
    nt.save(NP)

    rt = RuleTable(RP)
    rt.add("站", levels=5, weight=6.0, muscles="all")
    rt.save(RP)
    print(f"名字表 {len(nt)} 项、规则表 {len(rt)} 条 → 已存盘")

    # 新脑子：只读两个文件
    b2 = build(sp, verbose=False)
    nt2 = NameTable.load(NP)
    rt2 = RuleTable.load(RP)
    n = rt2.install(b2.engine, b2, nt2)
    cue = nt2.get("站")
    b2.engine.act[:] = 0.0
    b2.engine.reset_adaptation()
    for _ in range(12):
        b2.engine.clear_external()
        b2.engine.set_input(np.zeros(b2.encoder.n_in, np.float32))
        b2.engine.set_external(cue, np.ones(len(cue), np.float32))
        b2.engine.step()
    f = b2.forces()
    print(f"重装 {n} 条边：点「站」力度 {f.mean():.2f}（期望 0.5）")
    os.remove(NP); os.remove(RP)
    print(f"结论：{'规则表可用（名字→动作，存盘重装即可）' if abs(f.mean() - 0.5) < 0.12 else '不对'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
