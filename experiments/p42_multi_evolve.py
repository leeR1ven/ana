"""多技能演化 + 融合：一次训多项动作（走路/转向/起身/够高），并做参数平均融合。

用法（服务器，需 mujoco）：
    python experiments/p42_multi_evolve.py --gens 16 --pop 26 --jobs 26
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

SKILLS = ("gait", "walk0", "turn1", "turn2", "getup2", "reach", "orbit")
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
    """参数平均融合（不是交换）：两个好模型取均值，相当于"求同存异"。"""
    out = {}
    for k, v in a.items():
        if k == "seed":
            out[k] = int(rng.integers(1, 10 ** 6))
        elif isinstance(v, float):
            out[k] = float((v + float(b.get(k, v))) / 2.0)
        else:
            out[k] = int(round((float(v) + float(b.get(k, v))) / 2.0))
    return out


def _score_skill(r):
    """单项技能得分：位移为主，姿态高度/相位/翻倒做修正。"""
    dx = float(r.get("dx", 0.0) or 0.0)
    sc = 2.0 * dx
    if "diag" in r:
        sc -= 0.3 * float(r["diag"])
    if r.get("maxh", 0.3) > 0.45:
        sc -= 1.0
    if r.get("fallen"):
        sc -= 1.0
    if "hmean" in r and r["hmean"]:
        sc -= 1.5 * abs(float(r["hmean"]) - 0.27)
    return sc


def _work(args):
    kw, ticks = args
    from agi.dog_tasks import run_skill
    total, detail = 0.0, {}
    for s in SKILLS:
        try:
            r = run_skill(kw, s, ticks)
        except Exception:
            detail[s] = None
            continue
        detail[s] = {"ok": bool(r.get("ok")), "dx": float(r.get("dx", 0) or 0)}
        total += _score_skill(r)
    return kw, total / max(1, len(SKILLS)), detail


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=14)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--jobs", type=int, default=0)
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--keep", type=int, default=3)
    a = p.parse_args(argv)
    jobs = a.jobs or max(1, (os.cpu_count() or 4) - 1)
    rng = np.random.default_rng(777)

    seed = json.load(open(os.path.join(ROOT, "gait_seed2.json"), encoding="utf-8"))["kw"]
    elite = [(seed, -9.0)]
    print("种子:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in seed.items()})

    for g in range(a.gens):
        scale = 1.0 if g < a.gens // 2 else 0.5
        cands = [mutate(elite[0][0], rng, scale) for _ in range(a.pop)]
        # 融合：前两名取参数平均，产生"集两家之长"的孩子
        if len(elite) >= 2:
            for _ in range(max(2, a.pop // 6)):
                cands.append(fuse(elite[0][0], elite[1][0], rng))
        cands.extend(kw for kw, _ in elite)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            outs = list(ex.map(_work, [(k, a.ticks) for k in cands]))
        outs.sort(key=lambda x: -x[1])
        elite = []
        for kw, sc, det in outs:
            if all(_far(kw, e[0]) for e in elite):
                elite.append((kw, sc))
            if len(elite) >= a.keep:
                break
        kw, sc, det = outs[0]
        with open(os.path.join(ROOT, "multi_evolved.json"), "w", encoding="utf-8") as f:
            json.dump({"score": sc, "kw": kw, "detail": det}, f, ensure_ascii=False, indent=2)
        print("第%2d代: score %.3f  %s" % (g + 1, sc,
              " ".join("%s:%s" % (s, ("✓" if (det.get(s) or {}).get("ok") else "✗"))
                       for s in SKILLS)), flush=True)
    print("完成，最好 score %.3f" % elite[0][1])
    return 0


def _far(a, b, thr=0.25):
    """两个模型参数差得够远才算不同的精英（避免精英全是同一个）。"""
    ka = np.array([a.get(k, 0.0) for k in FLOAT_KEYS], float)
    kb = np.array([b.get(k, 0.0) for k in FLOAT_KEYS], float)
    return float(np.abs(ka - kb).mean()) > thr * 0.05


if __name__ == "__main__":
    raise SystemExit(main())
