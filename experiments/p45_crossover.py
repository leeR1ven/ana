"""交叉融合：按"模块"整块交换参数（不是逐参数平均），避免破坏各自可工作的组合。

用法（服务器）：python experiments/p45_crossover.py
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SKILLS = ("gait", "walk0", "turn1", "turn2", "getup2", "reach", "orbit", "tilt",
          "follow", "gaze")
MODULES = {
    "cpg": ("cpg_units", "cpg_gain", "cpg_inh"),
    "brain": ("k_near", "region_links", "w_lo", "w_hi", "motor_fan_in"),
    "body": ("posture_offset", "force_fan_in"),
}


def _ev(kw):
    from agi.dog_tasks import run_skill
    passes = []
    for s in SKILLS:
        try:
            if run_skill(kw, s, 120).get("ok"):
                passes.append(s)
        except Exception:
            pass
    return passes


def _run_one(item):
    name, kw = item
    return name, _ev(kw)


def crossover(a, b, modules):
    """modules 里的几块参数整块取自 b，其余取自 a。"""
    out = dict(a)
    for m in modules:
        for k in MODULES[m]:
            if k in b:
                out[k] = b[k]
    return out


def main() -> int:
    A = json.load(open(os.path.join(ROOT, "multi2_best.json"), encoding="utf-8"))["kw"]
    B = json.load(open(os.path.join(ROOT, "gait_best.json"), encoding="utf-8"))["kw"]

    cands = [("A(multi2_best)", A), ("B(gait_best)", B)]
    n = 1
    for m in MODULES:
        cands.append(("A+cpg" if m == "cpg" else "A+%s" % m, crossover(A, B, (m,))))
        cands.append(("B+cpg" if m == "cpg" else "B+%s" % m, crossover(B, A, (m,))))
        n += 2
    # 两块一起换
    for m1, m2 in (("cpg", "body"), ("cpg", "brain")):
        cands.append(("A+%s+%s" % (m1, m2), crossover(A, B, (m1, m2))))
        cands.append(("B+%s+%s" % (m1, m2), crossover(B, A, (m1, m2))))

    with ProcessPoolExecutor(max_workers=min(20, len(cands))) as ex:
        res = list(ex.map(_run_one, cands))
    res.sort(key=lambda x: -len(x[1]))
    for name, passes in res:
        print("%-24s %d/%d  %s" % (name, len(passes), len(SKILLS), ",".join(passes)))
    best = res[0]
    print("\n最好: %s (%d/%d)" % (best[0], len(best[1]), len(SKILLS)))
    with open(os.path.join(ROOT, "crossover_result.json"), "w", encoding="utf-8") as f:
        json.dump([{"name": n, "passes": p} for n, p in res], f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
