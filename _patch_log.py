p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()
old = """        print("第%2d代: 最远 %.2f 米(倒=%s) | 不摔最远 %.2f 米"
              % (g + 1, best_any[1], best_any[2], best_nf[1]), flush=True)"""
new = """        ba = next(r for kw2, r in outs if kw2 is best_any[0])
        bn = next(r for kw2, r in outs if kw2 is best_nf[0])
        print("第%2d代: 最快 %.3f m/s (走 %.2f 米/%d 拍/倒=%s) | 不摔最快 %.3f m/s (%.2f 米)"
              % (g + 1, ba["dx"] / max(1, ba["ticks"]) * 50.0, ba["dx"], ba["ticks"], ba["fell"],
                 bn["dx"] / max(1, bn["ticks"]) * 50.0, bn["dx"]), flush=True)"""
assert old in s, "anchor missing"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("logging fixed")
