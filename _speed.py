import json
import sys

sys.path.insert(0, "/root/ana")
from experiments.p49_walk_far import walk

kw = json.load(open("/root/ana/walk_far.json", encoding="utf-8"))["kw"]
r = walk(kw)
print("位移 %.2f 米 | 拍数 %d | 速度 %.3f 米/秒 | 倒=%s"
      % (r["dx"], r["ticks"], r["dx"] / max(1, r["ticks"]) * 50.0, r["fell"]))
