p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()
old = """        scored = []
        for kw, r in outs:
            v = (r["dx"] / max(1, r["ticks"])) * 50.0
            scored.append((kw, v if r["ticks"] >= 300 else -1.0, r["fell"]))"""
new = """        scored = []
        for kw, r in outs:
            v = (r["dx"] / max(1, r["ticks"])) * 50.0
            # 速度下限：低于 0.1 m/s 视为"没在走"，直接 0 分。
            # 否则"站着不动 697 拍"也能拿到 0.036 的分，等于奖励偷懒。
            if r["ticks"] < 300 or v < 0.10:
                v = -1.0
            scored.append((kw, v, r["fell"]))"""
assert old in s, "anchor missing"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("speed floor added")
