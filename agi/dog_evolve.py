# -*- coding: utf-8 -*-
"""狗技能演化：全量测试每个候选，技能只增不减。

规则（写死在筛选里）：
  · 每个候选都跑**全部**技能，指标全存档；
  · 技能清单只增不减；已获得技能退化 → 直接淘汰；
  · 每项技能保留前 2 名（专科池），与综合第一一起繁育；
  · 融合 = 出生参数按位平均（seed 不融合），随后变异。

用法（系统 Python，需 mujoco）：
    ANA_GO2_XML=/root/go2models/scene.xml python -m agi.dog_evolve \
        --gens 5 --pop 8 --workers 8 --init walk_best.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi.dog_tasks import SKILLS, run_skill          # noqa: E402

PARAMS = ("k_near", "region_links", "w_lo", "w_hi", "motor_fan_in",
          "posture_offset", "clock_motor", "torso_force",
          "wall_inh_gain", "clock_slope", "clock_rand")
RANGES = {"k_near": (3, 40), "region_links": (0, 256), "w_lo": (0.02, 0.5),
          "w_hi": (0.3, 2.0), "motor_fan_in": (4, 64),
          "posture_offset": (0.12, 1.4), "clock_motor": (0.0, 0.9), "torso_force": (0.0, 1.0), "wall_inh_gain": (0.3, 2.0), "clock_slope": (2, 14), "clock_rand": (0.0, 0.6)}


def random_kw(rng):
    return dict(seed=int(rng.integers(1, 10 ** 6)),
                k_near=int(rng.integers(3, 41)),
                region_links=int(rng.integers(0, 257)),
                w_lo=float(rng.uniform(0.02, 0.5)),
                w_hi=float(rng.uniform(0.3, 2.0)),
                motor_fan_in=int(rng.integers(4, 65)),
                posture_offset=float(rng.uniform(0.12, 0.6)),
                clock_motor=float(rng.uniform(0.0, 0.3)),
                torso_force=float(rng.uniform(0.0, 1.0)),
                wall_inh=True, wall_inh_gain=float(rng.uniform(0.5, 2.0)),
                clock_gait=True, clock_knee=0.5,
                clock_slope=int(rng.integers(2, 15)),
                clock_rand=float(rng.uniform(0.0, 0.4)))


def mutate(kw, rng):
    out = dict(kw)
    out["seed"] = int(rng.integers(1, 10 ** 6))
    out["k_near"] = int(np.clip(out.get("k_near", 8) + rng.integers(-3, 4), 3, 40))
    out["region_links"] = int(np.clip(out.get("region_links", 64) + rng.integers(-32, 33), 0, 256))
    for k in ("w_lo", "w_hi"):
        lo, hi = RANGES[k]
        out[k] = float(np.clip(out.get(k, (lo + hi) / 2) * rng.uniform(0.8, 1.25), lo, hi))
    out["motor_fan_in"] = int(np.clip(out.get("motor_fan_in", 8) * rng.uniform(0.6, 1.7), 4, 64))
    out["posture_offset"] = float(np.clip(out.get("posture_offset", 0.12) * rng.uniform(0.85, 1.25), 0.12, 1.4))
    if rng.random() < 0.15 or out.get("clock_motor", 0.0) == 0.0:
        out["clock_motor"] = float(np.clip(rng.uniform(0.0, 0.3), 0.0, 0.6))
    else:
        out["clock_motor"] = float(np.clip(out["clock_motor"] * rng.uniform(0.8, 1.3), 0.0, 0.9))
    out["torso_force"] = float(np.clip(out.get("torso_force", 0.0) * rng.uniform(0.8, 1.25), 0.0, 1.0))
    out["clock_gait"] = True
    out["clock_knee"] = 0.5
    out["clock_slope"] = int(np.clip(out.get("clock_slope", 3) + rng.integers(-2, 3), 2, 14))
    out["clock_rand"] = float(np.clip(out.get("clock_rand", 0.0) * rng.uniform(0.7, 1.4), 0.0, 0.6))
    out["wall_inh"] = True
    out["wall_inh_gain"] = float(np.clip(out.get("wall_inh_gain", 1.0) * rng.uniform(0.8, 1.25), 0.3, 2.0))
    return out


def fuse(a, b):
    """出生参数按位平均（seed 不融合）。"""
    out = {"seed": int(np.random.default_rng(0).integers(1, 10 ** 6))}
    for k in PARAMS:
        va = float(a.get(k, b.get(k, RANGES[k][0])))
        vb = float(b.get(k, va))
        out[k] = (va + vb) / 2.0
    for k in ("k_near", "region_links", "motor_fan_in"):
        out[k] = int(round(out[k]))
    return out


def _eval_candidate(args):
    """一个候选：建一次脑，跑完全部技能。"""
    kw, ticks = args
    from agi.dog_tasks import _brain
    try:
        brain = _brain(kw)[1]
    except Exception:
        return kw, {s: (False, 0.0) for s in SKILLS}
    out = {}
    for s in SKILLS:
        try:
            r = run_skill(kw, s, ticks, brain=brain)
        except Exception:
            r = {"ok": False}
        metric = abs(r.get("dx", r.get("dyaw", r.get("height", 0.0))))
        out[s] = (bool(r.get("ok")), float(metric))
    return kw, out


def evaluate_many(cands, ticks=80, workers=1):
    if workers <= 1 or len(cands) <= 1:
        return [_eval_candidate((kw, ticks))[1] for kw in cands]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(workers, len(cands))) as ex:
        outs = list(ex.map(_eval_candidate, [(kw, ticks) for kw in cands]))
    return [o[1] for o in outs]


# 每项技能的达标线：用来把"离通过多远"归一化，避免某些技能因量纲大而主导
NEEDS = {"walk0": 0.015, "walk1": 0.03, "walk2": 0.05, "walk3": 0.10,
         "walk4": 0.20, "walk5": 0.30, "walk_to_ball": 0.03,
         "turn1": 0.05, "turn2": 0.15, "turn3": 0.30,
         "getup0": 0.15, "getup1": 0.18, "getup2": 0.22, "getup3": 0.35,
         "follow": 0.004, "follow2": 0.010,
         "obstacle": 0.02, "obstacle2": 0.02, "march": 0.03,
         "back": 0.03, "tilt": 0.25, "lie": 0.12,
         "spin": 6.28, "crawl": 0.05, "shake": 0.01,
         "jump": 0.10, "detour": 0.03, "chase": 0.02, "gaze": 0.10,
         "lift": 5.0, "orient": 0.03, "robust": 0.05, "recover": 0.22,
         "verge": 0.01, "goto": 0.18}


def rank_key(scores, ledger):
    broke = any(ledger.get(s) and not scores[s][0] for s in ledger)
    earned = sum(1 for ok, _ in scores.values() if ok)
    # 归一化进度：每项 metric/门槛（封顶 1.2），这样 getup 的 0.155 也有一票
    prog = 0.0
    for s, (ok, m) in scores.items():
        need = NEEDS.get(s, 0.05)
        prog += 1.2 if ok else min(m / max(1e-6, need), 1.2)
    return (0 if broke else 1, earned, round(prog, 4))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=3)
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--ticks", type=int, default=80)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--init", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="dog_evolve_result.json")
    a = p.parse_args(argv)
    rng = np.random.default_rng(2024 + a.seed)

    ledger: dict[str, int] = {}
    best = None
    specials: dict[str, list] = {}
    history = []

    if a.init and os.path.exists(a.init):
        with open(a.init, encoding="utf-8") as f:
            kw0 = json.load(f)["kw"]
        sc = _eval_candidate((kw0, a.ticks))[1]
        best = (rank_key(sc, ledger), kw0, sc)
        for s, (ok, m) in sc.items():
            if ok:
                ledger[s] = 0
                specials.setdefault(s, []).append((m, kw0))
        print("从已有模型起步：", len(ledger), "项", list(ledger), flush=True)

    for g in range(a.gens):
        if best is None:
            cands = [random_kw(rng) for _ in range(a.pop)]
        else:
            cands = [mutate(best[1], rng)]
            for s, lst in specials.items():
                if not lst:
                    continue
                top = max(lst, key=lambda x: x[0])[1]
                cands.append(mutate(top, rng))
                cands.append(mutate(fuse(best[1], top), rng))
            while len(cands) < a.pop:
                cands.append(mutate(best[1], rng))
            cands = cands[:a.pop]

        scores_all = evaluate_many(cands, ticks=a.ticks, workers=a.workers)
        rows = [(rank_key(sc, ledger), kw, sc) for kw, sc in zip(cands, scores_all)]
        for rk, kw, sc in rows:
            got = [s for s, (ok, _) in sc.items() if ok]
            print(f"  代{g+1} {rk[:2]} 通过 {got}", flush=True)
        rows.sort(key=lambda r: r[0], reverse=True)
        top = rows[0]
        if best is None or top[0] > best[0]:
            best = (top[0], top[1], top[2])

        for rk, kw, sc in rows:
            for s, (ok, m) in sc.items():
                if not ok:
                    continue
                lst = specials.setdefault(s, [])
                lst.append((m, kw))
                lst.sort(key=lambda x: -x[0])
                del lst[2:]
        for s, (ok, _) in top[2].items():
            if ok and s not in ledger:
                ledger[s] = g + 1
                print(f"  ★ 新技能 {s}（第 {g+1} 代）", flush=True)

        history.append({"gen": g + 1, "skills": dict(ledger),
                        "best_rank": list(best[0][:2])})
        print(f"第 {g+1} 代：技能 {len(ledger)}/{len(SKILLS)} {list(ledger)}", flush=True)

    best_by_skill = {s: round(max(x[0] for x in lst), 4) for s, lst in specials.items() if lst}
    out = {"skills": ledger, "best_by_skill": best_by_skill,
           "best_kw": best[1] if best else None, "history": history}
    path = os.path.join(ROOT, a.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("技能清单:", json.dumps(ledger, ensure_ascii=False))
    print("各项最好指标:", json.dumps(best_by_skill, ensure_ascii=False))
    print("已写", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
