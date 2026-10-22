"""逐项核验 + 渲染帧：把每个技能的细节指标打出来，并出几张图。

用法：python experiments/p38_verify.py [kw.json] [技能名...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    src = argv[0] if argv else os.path.join(ROOT, "walk_seed3.json")
    only = argv[1:]
    kw = json.load(open(src, encoding="utf-8"))["kw"]

    from agi.dog_tasks import SKILLS, run_skill
    from agi.body_go2 import Go2Body
    from agi.dog_tasks import _ball, _brain, _obs

    print("模型:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items()})
    print("%-9s %-5s %-9s %-8s %-7s %-7s" % ("技能", "通过", "位移", "最高", "对角相关", "幅度"))
    bad = []
    for s in (only or SKILLS):
        r = run_skill(kw, s, 120)
        dx = r.get("dx", r.get("yaw", r.get("height", 0.0)))
        print("%-9s %-5s %-9.3f %-8s %-7s %-7s" %
              (s, "✓" if r.get("ok") else "✗", float(dx),
               round(float(r.get("maxh", 0)), 3) if "maxh" in r else "-",
               round(float(r.get("diag", 0)), 2) if "diag" in r else "-",
               round(float(r.get("amp", 0)), 1) if "amp" in r else "-"))
        if r.get("ok") and ("maxh" in r and r["maxh"] > 0.45):
            bad.append((s, "飞了", r["maxh"]))
        if r.get("ok") and ("diag" in r and r["diag"] > 0.5):
            bad.append((s, "腿同相", r["diag"]))

    if bad:
        print("\n⚠ 可疑的通过项：")
        for s, why, v in bad:
            print("  %s: %s（%.2f）" % (s, why, v))
    else:
        print("\n没有发现可疑通过项（仍建议看图）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
