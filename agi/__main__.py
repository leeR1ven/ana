# -*- coding: utf-8 -*-
"""命令行：建网 / 自检 / 基准 / 闭环演示。

    python -m agi info --preset full
    python -m agi check --preset full
    python -m agi bench --preset full
    python -m agi demo --preset full --ticks 400
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .bench import bench_engine, report
from .build import build
from .check import check, check_behavior
from .loop import Cortex
from .spec import PRESETS, build_spec
from .world import VirtualDog


def _brain(args):
    sp = build_spec(args.preset, **({"substeps": args.substeps} if args.substeps else {}))
    t0 = time.perf_counter()
    b = build(sp, verbose=args.verbose, do_calibrate=not args.no_calibrate)
    if args.verbose:
        print(f"[build] {time.perf_counter() - t0:.2f}s")
    return b


def cmd_info(args):
    sp = build_spec(args.preset)
    est = sp.estimate()
    print(f"预设 {args.preset}: {sp.px}×{sp.px}×3 视觉 + {sp.audio_bands}×{sp.audio_frames} 听觉"
          f" + 触觉 {sp.touch_ch} + 本体 {sp.proprio_ch} + 平衡 {sp.balance_ch}")
    for k, v in est.items():
        print(f"  {k:16s} {v:,}")
    print(f"  层计划：{len(sp.layers())} 层")
    for L in sp.layers()[:6]:
        print(f"    {L}")


def cmd_build(args):
    b = _brain(args)
    print(b.summary())


def cmd_check(args):
    b = _brain(args)
    issues = check(b, verbose=True)
    if not args.no_behavior:
        issues += check_behavior(b)
    print("通过" if not issues else f"失败 {len(issues)} 条")
    return 0 if not issues else 1


def cmd_bench(args):
    b = _brain(args)
    print(report(b, ticks=args.ticks))
    from .loop import Cortex
    c = Cortex(b, trace=False).bench(ticks=args.ticks)
    print(f"整拍（编码+步进+读）mean {c['mean_ms']:.2f} ms  p95 {c['p95_ms']:.2f} ms  "
          f"预算 {c['budget_ms']:.1f} ms")


def cmd_demo(args):
    b = _brain(args)
    dog = VirtualDog(b.spec, seed=args.seed)
    ctx = Cortex(b, body=dog)
    obs = dog.observe()
    t0 = time.perf_counter()
    n = args.ticks
    for i in range(n):
        tk = ctx.tick(obs)
        dog.step(tk.forces)
        obs = dog.observe()
        if args.verbose and (i % max(1, n // 10) == 0 or i == n - 1):
            d = float(np.hypot(*(dog.target - np.array([dog.x, dog.y]))))
            print(f"  t={i:4d} 前额叶 {tk.pfc_active:5d}  力 {np.round(tk.forces, 1)}  "
                  f"距离 {d:.2f}  朝向 {dog.th:+.2f}  {tk.ms['total']:.1f} ms")
    wall = time.perf_counter() - t0
    d = float(np.hypot(*(dog.target - np.array([dog.x, dog.y]))))
    tr = ctx.trace.arrays()
    print(f"\n跑了 {n} 拍 / {wall:.2f}s（{n / wall:.1f} 拍/秒，实时需要 "
          f"{b.spec.rate_hz:.0f}）")
    print(f"最终位置 ({dog.x:+.2f},{dog.y:+.2f})  离目标 {d:.2f}  朝向 {dog.th:+.2f}")
    print(f"出力均值 {tr['forces'].mean():.2f}  前额叶活跃 "
          f"{tr['pfc_active'].mean():.0f}  "
          f"整拍 {tr['ms_total'].mean():.2f} ms（p95 "
          f"{np.percentile(tr['ms_total'], 95):.2f}）")
    if args.save:
        ctx.trace.save(args.save)
        print(f"遥测已存 {args.save}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agi", description="关联神经元架构（v0 空框架）")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (("info", cmd_info, "打印规模估计"),
                            ("build", cmd_build, "建网并打印摘要"),
                            ("check", cmd_check, "架构自检"),
                            ("bench", cmd_bench, "耗时基准"),
                            ("demo", cmd_demo, "虚拟身体闭环演示")):
        q = sub.add_parser(name, help=help_)
        q.add_argument("--preset", default="xl", choices=sorted(PRESETS))
        q.add_argument("--verbose", action="store_true")
        q.add_argument("--substeps", type=int, default=0)
        q.add_argument("--no-calibrate", action="store_true")
        if name in ("check", "bench", "demo"):
            q.add_argument("--ticks", type=int, default=200)
            q.add_argument("--no-behavior", action="store_true")
        if name == "demo":
            q.add_argument("--seed", type=int, default=0)
            q.add_argument("--save", default="")
        q.set_defaults(func=fn)
    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
