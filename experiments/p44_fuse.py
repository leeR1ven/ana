"""名人堂 + 交叉融合：评估一批种子，两两求参数平均，看融合后代能不能兼得长处。

用法（服务器）：python experiments/p44_fuse.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SKILLS = ("gait", "walk0", "turn1", "turn2", "getup2", "reach", "orbit", "tilt",
          "follow", "gaze")


def _ev(kw):
    from agi.dog_tasks import run_skill
    out, passes = {}, []
    for s in SKILLS:
        try:
            r = run_skill(kw, s, 120)
        except Exception:
            out[s] = False
            continue
        ok = bool(r.get("ok"))
        out[s] = ok
        if ok:
            passes.append(s)
    return passes, out


def fuse(a, b):
    out = {}
    for k, v in a.items():
        if k == "seed":
            out[k] = int((int(a.get("seed", 1)) + int(b.get("seed", 2))) // 2 + 1)
        elif isinstance(v, float):
            out[k] = (v + float(b.get(k, v))) / 2.0
        else:
            out[k] = int(round((float(v) + float(b.get(k, v))) / 2.0))
    return out


def _run_one(item):
    name, kw = item
    return name, _ev(kw)


def main() -> int:
    seeds = []
    for name in ("gait_seed2.json", "gait_seed3.json", "multi2_best.json",
                 "gait_top1.json", "gait_best.json"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            try:
                seeds.append((name, json.load(open(p, encoding="utf-8"))["kw"]))
            except Exception:
                pass
    print("种子数:", len(seeds), [n for n, _ in seeds])

    jobs = [(n, kw) for n, kw in seeds]
    fused = []
    for (na, ka), (nb, kb) in itertools.combinations(seeds, 2):
        fused.append(("%s×%s" % (na.replace(".json", ""), nb.replace(".json", "")),
                      fuse(ka, kb)))
    print("融合组合:", len(fused))

    allc = jobs + fused
    with ProcessPoolExecutor(max_workers=min(24, len(allc))) as ex:
        res = list(ex.map(_run_one, allc))

    best = None
    for name, (passes, detail) in res:
        print("%-28s %d/%d  %s" % (name, len(passes), len(SKILLS), ",".join(passes)))
        if best is None or len(passes) > len(best[1]):
            best = (name, passes, detail)
    print("\n最好: %s 通过 %d 项" % (best[0], len(best[1])))
    with open(os.path.join(ROOT, "hall_of_fame.json"), "w", encoding="utf-8") as f:
        json.dump([{"name": n, "passes": p, "detail": d} for n, (p, d) in res],
                  f, ensure_ascii=False, indent=2)
    print("已写 hall_of_fame.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
