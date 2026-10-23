p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()
old = '        scored = [(kw, r["dx"], r["fell"]) for kw, r in outs]'
new = "        # 比速度（米/秒）：不然\"慢慢挪十分钟\"也能赢。\n"
new += "        # 存活太短（<300 拍）不计，避免\"冲一下就倒\"刷出假速度。\n"
new += "        scored = []\n"
new += "        for kw, r in outs:\n"
new += "            v = (r[\"dx\"] / max(1, r[\"ticks\"])) * 50.0\n"
new += "            scored.append((kw, v if r[\"ticks\"] >= 300 else -1.0, r[\"fell\"]))\n"
assert old in s, "not found"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("score = speed ok")
