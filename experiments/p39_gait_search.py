"""大批量找"会走"的狗：随机生成出生参数，用 gait 判据筛选。

用法（系统 Python，需 mujoco）：
    python experiments/p39_gait_search.py --n 300 --jobs 16
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


def _work(args):
    kw, ticks = args
    from agi.dog_tasks import run_skill
    try:
        r = run_skill(kw, "gait", ticks)
    except Exception:
        return kw, dict(ok=False, diag=1.0, dx=0.0, amp=0.0, maxh=9.0)
    return kw, r


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--jobs", type=int, default=0)
    p.add_argument("--ticks", type=int, default=150)
    p.add_argument("--seed", type=int, default=5)
    a = p.parse_args(argv)
    jobs = a.jobs or max(1, (os.cpu_count() or 4) - 1)
    rng = np.random.default_rng(a.seed)

    kws = []
    for _ in range(a.n):
        kws.append(dict(
            seed=int(rng.integers(1, 10 ** 6)),
            k_near=int(rng.integers(3, 40)),
            region_links=int(rng.integers(0, 256)),
            w_lo=float(rng.uniform(0.01, 1.0)),
            w_hi=float(rng.uniform(0.3, 6.0)),
            motor_fan_in=int(rng.integers(4, 64)),
            posture_offset=float(rng.uniform(0.12, 1.2)),
            cpg_units=int(rng.choice([0, 2, 4, 8, 12])),
            cpg_gain=float(rng.uniform(0.5, 12.0)),
            cpg_inh=float(rng.uniform(0.1, 2.0)),
            force_fan_in=int(rng.choice([0, 4, 8, 16])),
        ))

    best = []
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for kw, r in ex.map(_work, [(k, a.ticks) for k in kws], chunksize=1):
            done += 1
            if r.get("ok"):
                best.append((r["diag"], r["dx"], kw))
                best.sort(key=lambda x: (x[0], -x[1]))
                best = best[:10]
                print("  #%d 会走！diag %.2f dx %.3f（累计 %d 只）"
                      % (done, r["diag"], r["dx"], len(best)), flush=True)
                # 立刻落盘，避免跑完才写（进程被杀就全丢）
                with open(os.path.join(ROOT, "gait_best.json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"diag": best[0][0], "dx": best[0][1],
                               "count": len(best), "kw": best[0][2]},
                              f, ensure_ascii=False, indent=2)
                # 同时保留前 10 名（要逐个核验）
                with open(os.path.join(ROOT, "gait_top.json"), "w",
                          encoding="utf-8") as f:
                    json.dump([{"diag": d, "dx": dx, "kw": k}
                               for d, dx, k in best], f,
                              ensure_ascii=False, indent=2)
            elif done % 50 == 0:
                print("  #%d 还没有会走的" % done, flush=True)

    print("样本 %d，会走 %d 只" % (a.n, len(best)))
    for d, dx, kw in best[:5]:
        print("  diag %.2f dx %.3f  kw: %s" % (d, dx,
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items()}))
    if best:
        with open(os.path.join(ROOT, "gait_best.json"), "w", encoding="utf-8") as f:
            json.dump({"diag": best[0][0], "dx": best[0][1], "kw": best[0][2]},
                      f, ensure_ascii=False, indent=2)
        print("已写 gait_best.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
