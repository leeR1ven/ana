import json
import sys

import numpy as np

sys.path.insert(0, "/root/ana")
from agi.dog_tasks import _brain, _obs

base = json.load(open("/root/ana/walk_seed3.json", encoding="utf-8"))["kw"]
for inh in (1.4, 0.9, 0.6, 0.3):
    kw = dict(base); kw.update(cpg_units=8, cpg_inh=inh)
    sp, brain = _brain(kw)
    eng = brain.engine
    ids = np.asarray(brain.motor["cpg"], dtype=np.int64).ravel()
    rows = []
    for _ in range(80):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, None)))
        eng.step()
        rows.append(np.asarray(eng.act)[ids].copy())
    rows = np.stack(rows)
    peaks = [int(np.argmax(rows[:, u])) for u in range(8)]
    fired = int((rows.max(0) > 0.3).sum())
    print("抑制 %.1f: 点火单元 %d/8  各单元最亮时刻 %s" % (inh, fired, peaks))
