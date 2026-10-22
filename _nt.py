import json
import sys

import numpy as np

sys.path.insert(0, "/root/ana")
from agi.dog_tasks import _ball, _brain, _obs, run_skill

base = json.load(open("/root/ana/walk_seed3.json", encoding="utf-8"))["kw"]
for nt in (512, 64, 32):
    kw = dict(base); kw.update(n_time=nt, clock_gait=True, clock_motor=1.0)
    sp, brain = _brain(kw)
    eng = brain.engine
    mr1 = np.asarray(brain.motor["mr1"], dtype=np.int64).ravel()
    sub = brain.motor["sub"]
    rec = {m: [] for m in (0, 3)}
    for _ in range(120):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
        act = np.asarray(eng.act)
        for m in (0, 3):
            rec[m].append(float(act[mr1[sub[m]]].mean()))
    a = np.asarray(rec[0]) - np.mean(rec[0])
    b = np.asarray(rec[3]) - np.mean(rec[3])
    c = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    r = run_skill(kw, "gait", 120)
    print("n_time %3d: FL-FR 相关 %+.2f | gait diag %.2f amp %.1f dx %.3f"
          % (nt, c, r.get("diag", 0), r.get("amp", 0), r.get("dx", 0)))
