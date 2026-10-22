"""多技能演化（修正版）：精英跨代保留 + 每代只测 3 项（走路必测 + 轮换 2 项）。

用法（服务器）：python experiments/p43_multi2.py --gens 20 --pop 24 --jobs 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ROTATE = ("walk0", "turn1", "turn2", "getup2", "reach", "orbit", "back", "tilt",
          "follow", "gaze", "lift", "obstacle", "chase", "verge", "biped")
FLOAT_KEYS = ("w_lo", "w_hi", "posture_offset", "cpg_gain", "cpg_inh")


def mutate(kw, rng, scale=1.0):
    out = dict(kw)
    out["seed"] = int(rng.integers(1, 10 ** 6))
    out["k_near"] = int(np.clip(kw["k_near"] + rng.integers(-4, 5) * scale, 3, 40))
    out["region_links"] = int(np.clip(kw["region_links"] + rng.integers(-30, 31) * scale, 0, 256))
    out["motor_fan_in"] = int(np.clip(kw["motor_fan_in"] + rng.integers(-6, 7) * scale, 4, 64))
    out["cpg_units"] = int(np.clip(kw.get("cpg_units", 8) + rng.integers(-2, 3) * scale, 0, 16))
    out["force_fan_in"] = int(np.clip(kw.get("force_fan_in", 0) + rng.integers(-2, 3) * scale, 0, 24))
    for k in FLOAT_KEYS:
        v = float(kw.get(k, 0.5))
        out[k] = float(np.clip(v * (1.0 + 0.25 * scale * rng.uniform(-1, 1)), 0.01, 12.0))
    return out


def fuse(a, b, rng):
    out = {}
    for k, v in a.items():
        if k == "seed":
            out[k] = int(rng.integers(1, 10 ** 6))
        elif isinstance(v, float):
            out[k] = float((v + float(b.get(k, v))) / 2.0)
        else:
            out[k] = int(round((float(v) + float(b.get(k, v))) / 2.0))
    return out


def _score(r):
    """位移为主，其它只做**轻**修正。

    有融合时不该用重罚掐死多样性：爬着走但走得远也是本事，
    后代通过融合可能同时继承"走得远"和"站得直"。
    只保留两个必须的约束：不许飞（>0.45）、腿不能完全同相。
    """
    sc = 2.0 * float(r.get("dx", 0.0) or 0.0)
    if "diag" in r:
        sc -= 0.2 * max(0.0, float(r["diag"]) - 0.3)
    if r.get("maxh", 0.3) > 0.45:
        sc -= 0.5
    return sc


def _work(args):
    kw, skills, ticks = args
    from agi.dog_tasks import run_skill
    out = {}
    total = 0.0
    for s in skills:
        try:
            r = run_skill(kw, s, ticks)
        except Exception:
            out[s] = None
            continue
        out[s] = {"ok": bool(r.get("ok")), "dx": float(r.get("dx", 0) or 0)}
        total += _score(r)
    return kw, total / max(1, len(skills)), out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=20)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--jobs", type=int, default=0)
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--keep", type=int, default=5)
    a = p.parse_args(argv)
    jobs = a.jobs or max(1, (os.cpu_count() or 4) - 1)
    rng = np.random.default_rng(4242)

    seed = json.load(open(os.path.join(ROOT, "gait_seed2.json"), encoding="utf-8"))["kw"]
    elite = []          # [(kw, score, detail)] —— 跨代保留
    best_overall = -9e9
    print("种子:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in seed.items()})

    for g in range(a.gens):
        skills = ("gait",) + (ROTATE[(g * 2) % len(ROTATE)],
                              ROTATE[(g * 2 + 1) % len(ROTATE)])
        scale = 1.0 if g < a.gens // 2 else 0.5
        base = elite[0][0] if elite else seed
        cands = [mutate(base, rng, scale) for _ in range(a.pop)]
        if len(elite) >= 2:
            for _ in range(4):
                cands.append(fuse(elite[0][0], elite[1][0], rng))
        cands.extend(kw for kw, _, _ in elite)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            outs = list(ex.map(_work, [(k, skills, a.ticks) for k in cands]))
        outs.sort(key=lambda x: -x[1])

        # 精英池：跨代保留 + 这次的优胜者（按 score 合并去重）
        pool = [(kw, sc, det) for kw, sc, det in outs[: a.keep]] + elite
        pool.sort(key=lambda x: -x[1])
        elite = pool[: a.keep]

        kw, sc, det = outs[0]
        if sc > best_overall:
            best_overall = sc
            with open(os.path.join(ROOT, "multi2_best.json"), "w", encoding="utf-8") as f:
                json.dump({"score": sc, "skills": list(skills), "kw": kw, "detail": det},
                          f, ensure_ascii=False, indent=2)
        print("第%2d代 [%s] score %.3f  %s" % (
            g + 1, ",".join(skills), sc,
            " ".join("%s:%s" % (s, "✓" if (det.get(s) or {}).get("ok") else "✗")
                     for s in skills)), flush=True)
    print("完成，最高 %.3f" % best_overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
