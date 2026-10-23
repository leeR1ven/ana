import json
import sys

sys.path.insert(0, "/root/ana")
from experiments.p49_walk_far import walk

base = json.load(open("/root/ana/walk_far.json", encoding="utf-8"))["kw"]
print("B2F  gain | 速度(m/s) 距离 拍数 倒")
for b2f in (False, True):
    for g in (0.5, 1.5, 3.0):
        kw = dict(base)
        kw.update(balance_to_force=b2f, balance_gain=g,
                  balance_sectors=36, balance_per_sector=20, trot=True)
        r = walk(kw)
        print(" %-5s %.1f | %.3f  %.2f  %4d %s"
              % (b2f, g, r["dx"] / max(1, r["ticks"]) * 50.0, r["dx"], r["ticks"], r["fell"]))
