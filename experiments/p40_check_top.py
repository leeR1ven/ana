"""核验前 N 名候选：对每个候选跑**全部技能**，打印细节（在服务器上跑）。

用法（服务器，需 mujoco）：
    python experiments/p40_check_top.py --top 10 --ticks 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--ticks", type=int, default=120)
    a = p.parse_args(argv)

    from agi.dog_tasks import SKILLS, run_skill

    path = os.path.join(ROOT, "gait_top.json")
    if not os.path.exists(path):
        print("没有 gait_top.json")
        return 1
    cands = json.load(open(path, encoding="utf-8"))[: a.top]
    print("候选数:", len(cands))

    for i, c in enumerate(cands, 1):
        kw = c["kw"]
        print("\n=== 候选 %d  diag %.3f dx %.3f ===" % (i, c["diag"], c["dx"]))
        print("  kw:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items()})
        passed = []
        bad = []
        for s in SKILLS:
            r = run_skill(kw, s, a.ticks)
            if r.get("ok"):
                passed.append(s)
                if r.get("maxh", 0) > 0.45:
                    bad.append((s, "飞了", round(r["maxh"], 2)))
                if r.get("diag", 0) > 0.5 and s == "gait":
                    bad.append((s, "腿同相", round(r["diag"], 2)))
        print("  通过 %d/%d: %s" % (len(passed), len(SKILLS), passed))
        if bad:
            print("  ⚠ 可疑:", bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
