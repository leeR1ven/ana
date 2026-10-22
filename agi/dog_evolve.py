# -*- coding: utf-8 -*-
"""狗技能演化：扫描 → 全技能评分 → 选优 → 融合 → 变异 → 再扫描。

不遗忘的规则（写死在筛选里）：
  · 技能清单只增不减；
  · 候选必须通过清单里所有"已获得"的技能，任何一项退化就淘汰。

用法（系统 Python，需 mujoco）：
    python -m agi.dog_evolve --gens 2 --pop 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi.dog_tasks import SKILLS, run_skill   # noqa: E402

PARAMS = ("k_near", "region_links", "w_lo", "w_hi")
RANGES = {"k_near": (4, 24), "region_links": (0, 128),
          "w_lo": (0.02, 0.5), "w_hi": (0.3, 2.0)}


def random_kw(rng):
    return dict(seed=int(rng.integers(1, 10 ** 6)),
                k_near=int(rng.integers(4, 25)),
                region_links=int(rng.integers(0, 129)),
                w_lo=float(rng.uniform(0.02, 0.5)),
                w_hi=float(rng.uniform(0.3, 2.0)))


def mutate(kw, rng):
    out = dict(kw)
    out["seed"] = int(rng.integers(1, 10 ** 6))
    out["k_near"] = int(np.clip(out["k_near"] + rng.integers(-2, 3), 4, 24))
    out["region_links"] = int(np.clip(out["region_links"] + rng.integers(-16, 17), 0, 128))
    for k in ("w_lo", "w_hi"):
        lo, hi = RANGES[k]
        out[k] = float(np.clip(out[k] * rng.uniform(0.8, 1.25), lo, hi))
    return out


def fuse(a, b):
    """出生参数按位平均（seed 不融合）。"""
    out = {"seed": int(np.random.default_rng(0).integers(1, 10 ** 6))}
    for k in PARAMS:
        out[k] = (a[k] + b[k]) / 2.0
    out["k_near"] = int(round(out["k_near"]))
    out["region_links"] = int(round(out["region_links"]))
    return out


def evaluate(kw, skills=SKILLS, ticks=80):
    """返回 {skill: (ok, metric)}。"""
    out = {}
    for s in skills:
        try:
            r = run_skill(kw, s, ticks)
        except Exception as e:                     # 某道题炸了不能拖垮整轮
            r = {"ok": False, "err": f"{type(e).__name__}: {e}"}
        metric = abs(r.get("dx", r.get("dyaw", r.get("height", 0.0))))
        out[s] = (bool(r.get("ok")), float(metric))
    return out


def rank_key(scores, ledger):
    earned = sum(1 for s, (ok, _) in scores.items() if ok)
    # 已获得的技能只要退化就算不合格
    broke = any(ledger.get(s) and not scores[s][0] for s in ledger)
    metric = sum(m for _, m in scores.values())
    return (0 if broke else 1, earned, metric)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=2)
    p.add_argument("--pop", type=int, default=3)
    p.add_argument("--ticks", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = np.random.default_rng(2024 + a.seed)

    ledger: dict[str, int] = {}
    best = None
    history = []
    for g in range(a.gens):
        if best is None:
            cands = [random_kw(rng) for _ in range(a.pop)]
        else:
            kids = [mutate(best[1], rng)]
            if a.pop >= 3:
                kids.append(mutate(fuse(best[1], best[1]), rng))
            kids += [mutate(best[1], rng) for _ in range(max(0, a.pop - len(kids)))]
            cands = kids

        rows = []
        for kw in cands:
            sc = evaluate(kw, ticks=a.ticks)
            rows.append((rank_key(sc, ledger), kw, sc))
            got = [s for s, (ok, _) in sc.items() if ok]
            print(f"  代{g+1} 得分 {rank_key(sc, ledger)[:2]} 通过 {got}", flush=True)
        rows.sort(key=lambda r: r[0], reverse=True)
        top = rows[0]
        if best is None or top[0] > best[0]:
            best = (top[0], top[1], top[2])
        # 更新清单：新技能记上代数
        for s, (ok, _) in top[2].items():
            if ok and s not in ledger:
                ledger[s] = g + 1
        history.append({"gen": g + 1, "skills": dict(ledger),
                        "best": best[0][:2]})
        print(f"第 {g+1} 代：技能 {len(ledger)}/{len(SKILLS)} {list(ledger)}", flush=True)

    out = {"skills": ledger, "best_kw": best[1],
           "best_rank": list(best[0][:2]), "history": history}
    path = os.path.join(ROOT, "dog_evolve_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("结果:", json.dumps(out["skills"], ensure_ascii=False), "→", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
