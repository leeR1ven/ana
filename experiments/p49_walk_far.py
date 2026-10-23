"""只测走路：谁走得最远。双精英池：既保留"不摔最远"，也保留"偏科最远（哪怕会倒）"。

用法（服务器）：
    python experiments/p49_walk_far.py --gens 40 --pop 40 --jobs 30 --ticks 400
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


def walk(kw, ticks=400, stall=60, cap=6000):
    """一直走到倒下或原地不动为止——不死板截断。

    * 走一步算一步：只要有前进，就继续测（上限 cap 拍，防死循环）；
    * 摔倒即结束，记录走了多远；
    * 连续 stall 拍没有明显前进，视作"停下/打转"，也结束。
    """
    from agi.body_go2 import Go2Body
    from agi.dog_tasks import _ball, _brain, _drive_balance, _obs

    sp, brain = _brain(kw)
    eng = brain.engine
    body = Go2Body(posture_offset=float(kw.get("posture_offset", 0.35)))
    body.reset()
    body.data.qpos[2] = 0.27
    body.data.qpos[7:19] = np.array([0.0, 0.9, -1.8] * 4)
    body.mj.mj_forward(body.model, body.data)
    x0 = float(body.data.qpos[0])
    best_x, best_t = x0, 0
    idle = 0
    grounded = 0
    for t in range(cap):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        st = body.state()
        _drive_balance(eng, brain, float(st.get("roll", 0.0)), float(st.get("pitch", 0.0)),
                       int(kw.get("balance_sectors", 36)), int(kw.get("balance_per_sector", 20)))
        eng.step()
        body.step(brain.forces())
        if float(body.data.qpos[2]) <= 0.31:
            grounded += 1
        x = float(body.data.qpos[0])
        if x > best_x + 0.002:
            best_x, best_t, idle = x, t, 0
        else:
            idle += 1
        if body.fallen():
            return dict(dx=best_x - x0, fell=True, ticks=t + 1,
                        hmax=float(body.data.qpos[2]),
                        ground=grounded / max(1, t + 1))
        if idle >= stall:
            return dict(dx=best_x - x0, fell=False, ticks=t + 1,
                        hmax=float(body.data.qpos[2]),
                        ground=grounded / max(1, t + 1))
    return dict(dx=best_x - x0, fell=False, ticks=cap, hmax=0.0,
                ground=grounded / max(1, cap))


def _work(a):
    kw, ticks = a
    try:
        return kw, walk(kw, ticks)
    except Exception:
        return kw, None


def mutate(kw, rng, scale=1.0):
    out = dict(kw)
    out["seed"] = int(rng.integers(1, 10 ** 6))
    out["k_near"] = int(np.clip(kw["k_near"] + rng.integers(-4, 5) * scale, 3, 40))
    out["region_links"] = int(np.clip(kw["region_links"] + rng.integers(-30, 31) * scale, 0, 256))
    out["motor_fan_in"] = int(np.clip(kw["motor_fan_in"] + rng.integers(-6, 7) * scale, 4, 64))
    out["cpg_units"] = int(np.clip(kw.get("cpg_units", 8) + rng.integers(-2, 3) * scale, 4, 16))
    out["force_fan_in"] = int(np.clip(kw.get("force_fan_in", 0) + rng.integers(-2, 3) * scale, 0, 24))
    out["trot"] = True
    out["balance_to_force"] = True
    out["balance_sectors"] = 36
    out["balance_per_sector"] = 20
    out["balance_gain"] = float(np.clip(kw.get("balance_gain", 0.5)
                                        * (1 + 0.4 * scale * rng.uniform(-1, 1)), 0.05, 6.0))
    for k in ("w_lo", "w_hi", "posture_offset", "cpg_gain", "cpg_inh"):
        v = float(kw.get(k, 0.5))
        out[k] = float(np.clip(v * (1 + 0.25 * scale * rng.uniform(-1, 1)), 0.01, 12.0))
    return out


def random_kw(rng):
    return dict(
        seed=int(rng.integers(1, 10 ** 6)),
        k_near=int(rng.integers(3, 40)),
        region_links=int(rng.integers(0, 256)),
        w_lo=float(rng.uniform(0.01, 1.0)),
        w_hi=float(rng.uniform(0.3, 6.0)),
        motor_fan_in=int(rng.integers(4, 64)),
        posture_offset=float(rng.uniform(0.12, 1.2)),
        cpg_units=int(rng.choice([4, 6, 8, 12, 16])),
        cpg_gain=float(rng.uniform(0.5, 12.0)),
        cpg_inh=float(rng.uniform(0.1, 2.0)),
        force_fan_in=int(rng.choice([0, 2, 4, 8, 16])),
        trot=True,
        balance_to_force=True,
        balance_sectors=36, balance_per_sector=20,
        balance_gain=float(rng.uniform(0.2, 4.0)),
    )


def fuse(a, b, rng):
    out = {}
    for k, v in a.items():
        if k == "seed":
            out[k] = int(rng.integers(1, 10 ** 6))
        elif isinstance(v, float):
            out[k] = (v + float(b.get(k, v))) / 2.0
        else:
            out[k] = int(round((float(v) + float(b.get(k, v))) / 2.0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=400)
    a = ap.parse_args(argv)
    jobs = a.jobs or max(1, (os.cpu_count() or 4) - 2)
    rng = np.random.default_rng(2024)

    seed = json.load(open(os.path.join(ROOT, "champion.json"), encoding="utf-8"))["kw"]
    seed.update(balance_to_force=True, trot=True, balance_sectors=36, balance_per_sector=20,
                balance_gain=float(seed.get("balance_gain", 0.5)))
    elites = []
    record = 0.0
    record_nf = 0.0
    for g in range(a.gens):
        scale = 1.0 if g < a.gens // 2 else 0.7
        base = elites[0][0] if elites else seed
        cands = [mutate(base, rng, scale) for _ in range(a.pop // 2)]
        cands += [random_kw(rng) for _ in range(a.pop // 4)]
        for i in range(max(2, a.pop // 8)):
            if len(elites) >= 2:
                cands.append(fuse(elites[i % len(elites)][0],
                                  elites[(i + 1) % len(elites)][0], rng))
        cands += [kw for kw, _, _ in elites]

        with ProcessPoolExecutor(max_workers=jobs) as ex:
            outs = [x for x in ex.map(_work, [(k, a.ticks) for k in cands]) if x[1]]

        # 比速度（米/秒）：不然"慢慢挪十分钟"也能赢。
        # 存活太短（<300 拍）不计，避免"冲一下就倒"刷出假速度。
        # 不设任何门槛/及格线：只按速度排名取最好的。
        # 就算全场最快只有 0.02 m/s，也照选不误——排名永远存在，不需要"及格"。
        # 速度 × 贴地时间占比：飞起来的那些时间不给分。
        # 这是乘性权重，不是及格线——排名照样存在。
        scored = [(kw,
                   (r["dx"] / max(1, r["ticks"]) * 50.0) * (0.3 + 0.7 * r.get("ground", 1.0)),
                   r["fell"])
                  for kw, r in outs]

        by_nofall = sorted(scored, key=lambda x: (not x[2], x[1]), reverse=True)
        by_dx = sorted(scored, key=lambda x: x[1], reverse=True)
        elites = []
        for kw, dx, fell in by_nofall[:3] + by_dx[:3]:      # 双池：稳的 + 偏科的
            if all(kw is not e[0] for e in elites):
                elites.append((kw, dx, fell))
        best_any, best_nf = by_dx[0], by_nofall[0]
        # 分开存：会倒的最快 / 不摔的最快。之前只存"最快"，
        # 结果记录被会倒的那只覆盖，"不摔"的成果丢了。
        if best_any[1] > record:
            record = best_any[1]
            with open(os.path.join(ROOT, "walk_far.json"), "w", encoding="utf-8") as f:
                json.dump({"dx": best_any[1], "fell": best_any[2], "kw": best_any[0]},
                          f, ensure_ascii=False, indent=2)
        if (not best_nf[2]) and best_nf[1] > record_nf:
            record_nf = best_nf[1]
            with open(os.path.join(ROOT, "walk_far_nofall.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"dx": best_nf[1], "fell": False, "kw": best_nf[0]},
                          f, ensure_ascii=False, indent=2)
        ba = next(r for kw2, r in outs if kw2 is best_any[0])
        bn = next(r for kw2, r in outs if kw2 is best_nf[0])
        print("第%2d代: 最快 %.3f m/s (走 %.2f 米/%d 拍/倒=%s) | 不摔最快 %.3f m/s (%.2f 米)"
              % (g + 1, ba["dx"] / max(1, ba["ticks"]) * 50.0, ba["dx"], ba["ticks"], ba["fell"],
                 bn["dx"] / max(1, bn["ticks"]) * 50.0, bn["dx"]), flush=True)
    print("纪录:", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
