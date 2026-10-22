import json
import sys

sys.path.insert(0, "/root/ana")
from agi.dog_tasks import run_skill

kw = json.load(open("/root/ana/walk_seed3.json", encoding="utf-8"))["kw"]
r = run_skill(kw, "goto", 120)
print("goto ok %s dx %+.3f" % (r.get("ok"), r.get("dx", 0)))
