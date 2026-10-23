p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()
s = s.replace("            if r[\"ticks\"] < 300 or v < 0.10:",
              "            if r[\"ticks\"] < 300 or v < 0.02:")
open(p, "w", encoding="utf-8").write(s)
print("floor -> 0.02")
