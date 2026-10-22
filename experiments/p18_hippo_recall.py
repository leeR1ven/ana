import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agi import build_spec, build, Cortex
from experiments.p1_recall import scene_obs

sp = build_spec("full")
b = build(sp, verbose=False)
ctx = Cortex(b)
eng = b.engine
ids = np.array(sorted(np.concatenate(
    [l.ids for n, l in b.layers.items() if n.endswith(".L5")]).tolist()))


def code():
    a = np.array(eng.act).copy()
    return set(np.flatnonzero(a[ids] > 0.5).tolist())


def cue(slot, n=8):
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(n):
        eng.clear_external()
        eng.set_input(np.zeros(b.encoder.n_in, np.float32))
        eng.set_external(np.array([slot]), np.ones(1, np.float32))
        eng.step()
    return code()


# ---- 第一圈：看 A，记在第 8 格 ----
snaps = []
for _ in range(16):
    ctx.tick(scene_obs(sp, 1), record=False)
    snaps.append(code())
b.binder.flush()
slot = int(b.time_ring.ids[8])
A = snaps[8]
got = cue(slot)
print(f"记忆A: 当时 {len(A)}, 回忆 {len(got)}, 覆盖 {len(got & A) / max(1, len(A)):.2f}, "
      f"精确 {len(got & A) / max(1, len(got)):.2f}")

# ---- 第二圈：看 B 很久，第 8 格被重新写 ----
for _ in range(600):
    ctx.tick(scene_obs(sp, 2), record=False)
b.binder.flush()
B = code()
got2 = cue(slot)
print(f"隔一圈后再点同一格: 回忆 {len(got2)}, 命中A {len(got2 & A) / max(1, len(A)):.2f}, "
      f"命中B {len(got2 & B) / max(1, len(B)):.2f}")
