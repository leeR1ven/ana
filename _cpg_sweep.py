import json
import sys

sys.path.insert(0, "/root/ana")
from experiments.p49_walk_far import walk

base = json.load(open("/root/ana/walk_far.json", encoding="utf-8"))["kw"]
print("cpg_gain cpg_units force_fan_in | m/s  距离 拍数 倒")
for gain in (3.5, 6.0, 10.0, 15.0):
    for units in (12, 16):
        kw = dict(base)
        kw.update(cpg_gain=gain, cpg_units=units)
        r = walk(kw)
        print(" %6.1f %6d %6d | %.3f %.2f %4d %s"
              % (gain, units, kw.get("force_fan_in", 0),
                 r["dx"] / max(1, r["ticks"]) * 50.0, r["dx"], r["ticks"], r["fell"]))
