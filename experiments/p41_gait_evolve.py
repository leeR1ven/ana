"""以"会走但会栽"的模型为种子，做针对性演化：变异 -> 评分 -> 选优 -> 重复。

用法（服务器，需 mujoco）：
    python experiments/p41_gait_evolve.py --gens 15 --pop 24 --jobs 24
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

INT_KEYS = ("k_near", "region_links", "motor_fan_in", "cpg_units", "force_fan_in", "seed")
FLOAT_KEYS = ("w_lo", "w_hi", "posture_offset", "cpg_gain", "cpg_inh")


def mutate(kw, rng, scale=1.0):
    out = dict(kw)
    out["seed"] = int(rng.integers(1, 10 ** 6))
    out["k_near"] = int(np.clip(kw["k_near"] + rng.integers(-4, 5) * scale, 3, 40))
    out["region_links"] = int(np.clip(kw["region_links"] + rng.integers(-30, 31) * scale, 0, 256))
    out["motor_fan_in"] = int(np.clip(kw["motor_fan_in"] + rng.integers(-6, 7) * scale, 4, 64))
    out["cpg_units"] = int(np.clip(kw.get("cpg_units", 8) + rng.integers(-2, 3) * scale, 0, 16))
    out["force_fan_in"] = int(np.clip(kw.get("force_fan_in", 0) + rng.integers(-2, 3) * scale, 0, 24))
    for k in ("w_lo", "w_hi", "posture_offset", "cpg_gain", "cpg_inh"):
        v = float(kw.get(k, 0.5))
        out[k] = float(np.clip(v * (1.0 + 0.25 * scale * rng.uniform(-1, 1)), 0.01, 12.0))
    return out


def _work(args):
    kw, ticks = args
    from agi.dog_tasks import run_skill
    try:
        r = run_skill(kw, "gait", ticks)
    except Exception:
        return kw, -5.0, dict(diag=1.0, dx=0.0, maxh=9.0)
    ok = bool(r.get("ok"))
    # 位移是主目标（走多远就多少分）；相位/翻倒只做惩罚，不能给"通过"加固定分，
    # 否则"走 0.25 米但没倒"会压过"走 0.71 米但最后栽了"——那是错的方向。
    score = 2.0 * float(r.get("dx", 0.0)) - 0.3 * float(r.get("diag", 1.0))
    if r.get("maxh", 9.0) > 0.45:
        score -= 1.0
    if r.get("fallen"):
        score -= 1.0
    if not ok:
        score -= 0.2
    # 姿态高度：正常行走应接近 0.27 米。趴着滑（0.15）或站太高（0.34+）都扣分，
    # 否则演化会选出"低姿滑行"——指标过关但画面难看。
    hm = float(r.get("hmean", 0.27) or 0.27)
    score -= 2.0 * abs(hm - 0.27)
    return kw, score, r


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=12)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--jobs", type=int, default=0)
    p.add_argument("--ticks", type=int, default=150)
    a = p.parse_args(argv)
    jobs = a.jobs or max(1, (os.cpu_count() or 4) - 1)
    rng = np.random.default_rng(2024)

    seed_path = os.path.join(ROOT, "gait_top1.json")
    best = json.load(open(seed_path, encoding="utf-8"))["kw"]
    print("种子:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()})

    best_score = -9.0
    for g in range(a.gens):
        scale = 1.0 if g < a.gens // 2 else 0.5          # 前期大变异、后期精调
        cands = [mutate(best, rng, scale) for _ in range(a.pop)]
        cands.append(dict(best))
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            outs = list(ex.map(_work, [(k, a.ticks) for k in cands]))
        outs.sort(key=lambda x: -x[1])
        kw, sc, r = outs[0]
        if sc > best_score:
            best_score, best = sc, kw
            with open(os.path.join(ROOT, "gait_evolved.json"), "w", encoding="utf-8") as f:
                json.dump({"score": sc, "kw": kw,
                           "diag": r.get("diag"), "dx": r.get("dx"),
                           "maxh": r.get("maxh"), "ok": r.get("ok")},
                          f, ensure_ascii=False, indent=2)
        print("第%2d代: 最好 score %.3f (dx %.3f diag %.2f maxh %.3f ok %s)"
              % (g + 1, sc, r.get("dx", 0), r.get("diag", 0), r.get("maxh", 0), r.get("ok")),
              flush=True)
    print("最终最好 score %.3f" % best_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
