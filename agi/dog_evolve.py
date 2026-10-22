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

PARAMS = ("k_near", "region_links", "w_lo", "w_hi", "motor_fan_in")
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
    out["motor_fan_in"] = int(np.clip(out.get("motor_fan_in", 8) * rng.uniform(0.6, 1.7), 4, 64))
    for k in ("w_lo", "w_hi"):
        lo, hi = RANGES[k]
        out[k] = float(np.clip(out[k] * rng.uniform(0.8, 1.25), lo, hi))
    return out


def fuse(a, b):
    """出生参数按位平均（seed 不融合）。"""
    out = {"seed": int(np.random.default_rng(0).integers(1, 10 ** 6))}
    for k in PARAMS:
        va = a.get(k, b.get(k, RANGES.get(k, (0, 1))[0]))
        vb = b.get(k, va)
        out[k] = (va + vb) / 2.0
    out["k_near"] = int(round(out["k_near"]))
    out["region_links"] = int(round(out["region_links"]))
    out["motor_fan_in"] = int(round(out.get("motor_fan_in", 8)))
    return out


def _eval_one(args):
    kw, skill, ticks = args
    try:
        r = run_skill(kw, skill, ticks)
    except Exception as e:
        r = {"ok": False, "err": f"{type(e).__name__}: {e}"}
    metric = abs(r.get("dx", r.get("dyaw", r.get("height", 0.0))))
    return skill, bool(r.get("ok")), float(metric)


def evaluate(kw, skills=SKILLS, ticks=80, workers=1):
    """返回 {skill: (ok, metric)}；workers>1 时把各技能丢进进程池并行跑。"""
    if workers > 1 and len(skills) > 1:
        from concurrent.futures import ProcessPoolExecutor
        out = {}
        with ProcessPoolExecutor(max_workers=min(workers, len(skills))) as ex:
            for s, ok, m in ex.map(_eval_one, [(kw, s, ticks) for s in skills]):
                out[s] = (ok, m)
        return out
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
    p.add_argument("--init", default="", help="从已有模型参数起步（json，含 kw）")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)
    rng = np.random.default_rng(2024 + a.seed)

    ledger: dict[str, int] = {}
    best = None
    if a.init and os.path.exists(a.init):      # 从已知最好的模型起步
        with open(a.init, encoding="utf-8") as f:
            _kw = json.load(f)["kw"]
        _sc = evaluate(_kw, ticks=a.ticks)
        best = (rank_key(_sc, ledger), _kw, _sc)
        for s, (ok, _) in _sc.items():
            if ok:
                ledger[s] = 0
        print(f"从已有模型起步：{len(ledger)} 项技能 {list(ledger)}", flush=True)
    history = []
    specials: dict[str, tuple] = {}          # 每项技能的"专科状元"
    for g in range(a.gens):
        if best is None:
            cands = [random_kw(rng) for _ in range(a.pop)]
        else:
            # 综合状元（best）+ 各科状元（specials）一起繁育
            kids = [mutate(best[1], rng)]
            for s, sp in specials.items():
                kids.append(mutate(sp[1], rng))                 # 专科的变异体
                kids.append(mutate(fuse(best[1], sp[1]), rng))  # 综合×专科 融合
            kids += [mutate(best[1], rng)
                     for _ in range(max(0, a.pop - len(kids)))]
            cands = kids[:max(1, a.pop)]

        rows = []
        for kw in cands:
            sc = evaluate(kw, ticks=a.ticks, workers=a.workers)
            rows.append((rank_key(sc, ledger), kw, sc))
            got = [s for s, (ok, _) in sc.items() if ok]
            print(f"  代{g+1} 得分 {rank_key(sc, ledger)[:2]} 通过 {got}", flush=True)
        rows.sort(key=lambda r: r[0], reverse=True)
        top = rows[0]
        if best is None or top[0] > best[0]:
            best = (top[0], top[1], top[2])
        # 更新"各科状元"：每项技能保留指标最高的那个
        for rk, kw, sc in rows:
            for s, (ok, m) in sc.items():
                if not ok:
                    continue
                cur = specials.get(s)
                if cur is None or m > cur[2]:
                    specials[s] = (rk, kw, m, sc)
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
