import json
import sys

sys.path.insert(0, "/root/ana")
from experiments.p49_walk_far import walk

base = json.load(open("/root/ana/walk_far.json", encoding="utf-8"))["kw"]
base.update(cpg_gain=6.0, cpg_units=16, trot=True,
            balance_sectors=36, balance_per_sector=20)
print("b2f  balance_gain | m/s  距离 拍数 倒")
for b2f in (False, True):
    for bg in (0.5, 2.0, 5.0, 10.0):
        kw = dict(base)
        kw.update(balance_to_force=b2f, balance_gain=bg)
        r = walk(kw)
        print(" %-5s %5.1f | %.3f %.2f %4d %s"
              % (b2f, bg, r["dx"] / max(1, r["ticks"]) * 50.0, r["dx"], r["ticks"], r["fell"]))
