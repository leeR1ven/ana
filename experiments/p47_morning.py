"""明早验收用：对 champion（或指定模型）做全技能体检，输出可直接贴进报告的表。

用法（服务器）：
    python experiments/p47_morning.py                # 体检 champion.json
    python experiments/p47_morning.py 某个模型.json   # 体检指定模型
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SKILLS = ("gait", "walk0", "walk2", "walk4", "turn1", "turn2", "turn3",
          "getup0", "getup2", "reach", "orbit", "tilt", "follow", "gaze",
          "biped", "lift", "back", "obstacle", "robust", "recover")


def _one(item):
    path, s = item
    from agi.dog_tasks import run_skill
    try:
        kw = json.load(open(path, encoding="utf-8"))["kw"]
    except Exception:
        return s, None
    try:
        r = run_skill(kw, s, 120)
    except Exception:
        return s, None
    return s, r


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    path = os.path.join(ROOT, argv[0] if argv else "champion.json")
    if not os.path.exists(path):
        print("找不到:", path)
        return 1
    print("模型:", path)

    jobs = [(path, s) for s in SKILLS]
    with ProcessPoolExecutor(max_workers=min(20, len(jobs))) as ex:
        res = {}
        for s, r in ex.map(_one, jobs):
            res[s] = r

    print("\n%-9s %-4s %8s %8s %8s %8s %6s" %
          ("技能", "通过", "位移", "最高", "对角相关", "平均高", "摆动"))
    print("-" * 58)
    passed = []
    for s in SKILLS:
        r = res.get(s)
        if not isinstance(r, dict):
            print("%-9s %-4s  （执行失败）" % (s, "?"))
            continue
        ok = bool(r.get("ok"))
        if ok:
            passed.append(s)
        def _f(k):
            v = r.get(k)
            return "%.3f" % float(v) if isinstance(v, (int, float)) else "-"
        print("%-9s %-4s %8s %8s %8s %8s %6s" % (
            s, "✓" if ok else "✗", _f("dx"), _f("maxh"), _f("diag"),
            _f("hmean"), _f("swing")))
    print("-" * 58)
    print("通过 %d/%d: %s" % (len(passed), len(SKILLS), ",".join(passed)))

    out = os.path.join(ROOT, "morning_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": os.path.basename(path), "passed": passed,
                   "detail": {s: (res.get(s) if isinstance(res.get(s), dict) else None)
                              for s in SKILLS}}, f, ensure_ascii=False, indent=2)
    print("已写", os.path.basename(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
