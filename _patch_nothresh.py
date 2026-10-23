p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()
old_start = s.index("        scored = []")
old_end = s.index("        by_nofall = sorted(")
new = """        # 不设任何门槛/及格线：只按速度排名取最好的。
        # 就算全场最快只有 0.02 m/s，也照选不误——排名永远存在，不需要"及格"。
        scored = [(kw, r["dx"] / max(1, r["ticks"]) * 50.0, r["fell"])
                  for kw, r in outs]

"""
s = s[:old_start] + new + s[old_end:]
open(p, "w", encoding="utf-8").write(s)
print("no threshold")
