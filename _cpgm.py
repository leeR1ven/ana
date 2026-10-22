import json
import sys

import numpy as np

sys.path.insert(0, "/root/ana")
from agi.dog_tasks import _brain, _obs

kw = json.load(open("/root/ana/walk_seed3.json", encoding="utf-8"))["kw"]
for cu in (0, 8):
    k = dict(kw); k["cpg_units"] = cu
    sp, brain = _brain(k)
    eng = brain.engine
    ids = brain.motor.get("cpg")
    if ids is None:
        print("cpg_units %d: 没有 CPG" % cu)
        continue
    ids = np.asarray(ids, dtype=np.int64).ravel()
    rec = []
    for _ in range(120):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, None)))
        eng.step()
        rec.append(float(np.asarray(eng.act)[ids].mean()))
    rec = np.asarray(rec)
    print("cpg_units %d: 平均 %.2f 最小 %.2f 最大 %.2f 标准差 %.3f 首20拍 %s"
          % (cu, rec.mean(), rec.min(), rec.max(), rec.std(), np.round(rec[:20], 2)))
