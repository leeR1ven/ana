p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()

# 1) walk() 里统计贴地占比
s = s.replace("""    best_x, best_t = x0, 0
    idle = 0""",
"""    best_x, best_t = x0, 0
    idle = 0
    grounded = 0""")
s = s.replace("""        x = float(body.data.qpos[0])""",
"""        if float(body.data.qpos[2]) <= 0.31:
            grounded += 1
        x = float(body.data.qpos[0])""")
s = s.replace("""            return dict(dx=best_x - x0, fell=True, ticks=t + 1, hmax=float(body.data.qpos[2]))""",
"""            return dict(dx=best_x - x0, fell=True, ticks=t + 1,
                        hmax=float(body.data.qpos[2]),
                        ground=grounded / max(1, t + 1))""")
s = s.replace("""            return dict(dx=best_x - x0, fell=False, ticks=t + 1, hmax=float(body.data.qpos[2]))""",
"""            return dict(dx=best_x - x0, fell=False, ticks=t + 1,
                        hmax=float(body.data.qpos[2]),
                        ground=grounded / max(1, t + 1))""")
s = s.replace("""    return dict(dx=best_x - x0, fell=False, ticks=cap, hmax=0.0)""",
"""    return dict(dx=best_x - x0, fell=False, ticks=cap, hmax=0.0,
                ground=grounded / max(1, cap))""")

# 2) 评分：速度 × 贴地占比（飞起来的时间不算数，但不是硬门槛）
s = s.replace("""        scored = [(kw, r["dx"] / max(1, r["ticks"]) * 50.0, r["fell"])
                  for kw, r in outs]""",
"""        # 速度 × 贴地时间占比：飞起来的那些时间不给分。
        # 这是乘性权重，不是及格线——排名照样存在。
        scored = [(kw,
                   (r["dx"] / max(1, r["ticks"]) * 50.0) * (0.3 + 0.7 * r.get("ground", 1.0)),
                   r["fell"])
                  for kw, r in outs]""")
open(p, "w", encoding="utf-8").write(s)
print("grounded weighting added")
