import sys
sys.path.insert(0, ".")
import numpy as np
from agi.tasks import GazeWorld, GazeBrain
gb = GazeBrain(0)

def pol(off, pairs):
    gb.ctx.tick(gb._see(off), record=False)
    f = np.asarray(gb.b.forces(), np.float32)
    ax = f[pairs[0]] - f[pairs[1]]
    ay = f[pairs[2]] - f[pairs[3]]
    return np.clip(np.array([ax, ay], np.float32) * 2.0, -1, 1)

best = None
import itertools
for a, b, c, d in itertools.permutations(range(12), 4):
    ok = 0
    for s in range(4):
        w = GazeWorld(seed=s)
        o = w.reset()
        while not w.done():
            o = w.step(pol(o, (a, b, c, d)))
        ok += (abs(float(w.ball[0] - w.gaze[0])) < 0.12
               and abs(float(w.ball[1] - w.gaze[1])) < 0.12)
    if best is None or ok > best[0]:
        best = (ok, (a, b, c, d))
    if ok >= 4:
        break
print("最好组合", best[1], "成功", best[0], "/4")
