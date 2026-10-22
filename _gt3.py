import json
import sys

sys.path.insert(0, "/root/ana")
from agi.dog_tasks import run_skill

kw = json.load(open("/root/ana/walk_seed3.json", encoding="utf-8"))["kw"]
r = run_skill(kw, "gait", 150)
print("gait ok %s dx %.3f 最高 %.3f 对角相关 %.2f 幅度 %.1f"
      % (r.get("ok"), r.get("dx", 0), r.get("maxh", 0), r.get("diag", 0), r.get("amp", 0)))
