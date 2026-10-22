import json
import sys

sys.path.insert(0, "/root/ana")
from agi.dog_tasks import run_skill

kw = json.load(open("/root/ana/walk_seed.json", encoding="utf-8"))["kw"]
for flag, gain in ((False, 0.5), (True, 0.3), (True, 0.6)):
    k = dict(kw)
    k["wall_inh"] = flag
    k["wall_inh_gain"] = gain
    r = run_skill(k, "obstacle", 100)
    print("wall_inh=%s gain=%.1f: 无墙 %.3f 有墙 %.3f 比值 %.2f ok %s"
          % (flag, gain, r.get("dx_free", 0), r.get("dx_wall", 0),
             abs(r.get("dx_wall", 0)) / max(1e-6, abs(r.get("dx_free", 0))), r.get("ok")))
