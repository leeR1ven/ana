"""对每一代存档做 10 项全技能体检，找出最全面的那只。

用法（服务器）：python experiments/p46_audit_gens.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SKILLS = ("gait", "walk0", "turn1", "turn2", "getup2", "reach", "orbit", "tilt",
          "follow", "gaze", "biped", "lift")


def _one(item):
    path = item
    from agi.dog_tasks import run_skill
    try:
        kw = json.load(open(path, encoding="utf-8"))["kw"]
    except Exception:
        return path, []
    passes = []
    for s in SKILLS:
        try:
            if run_skill(kw, s, 120).get("ok"):
                passes.append(s)
        except Exception:
            pass
    return os.path.basename(path), passes


def main() -> int:
    files = sorted(glob.glob(os.path.join(ROOT, "multi2_g*.json")))
    extra = [os.path.join(ROOT, n) for n in ("multi3_best.json", "gait_best.json",
                                             "multi2_best.json")]
    files += [p for p in extra if os.path.exists(p)]
    print("待检查:", len(files))
    with ProcessPoolExecutor(max_workers=min(20, len(files))) as ex:
        res = list(ex.map(_one, files))
    res.sort(key=lambda x: -len(x[1]))
    for name, passes in res:
        print("%-24s %2d/%d  %s" % (name, len(passes), len(SKILLS), ",".join(passes)))
    if res:
        best = res[0]
        print("\n最全面: %s (%d/%d)" % (best[0], len(best[1]), len(SKILLS)))
        kw = json.load(open(os.path.join(ROOT, best[0]), encoding="utf-8"))["kw"]
        with open(os.path.join(ROOT, "champion.json"), "w", encoding="utf-8") as f:
            json.dump({"source": best[0], "passes": best[1], "kw": kw},
                      f, ensure_ascii=False, indent=2)
        print("已写 champion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
