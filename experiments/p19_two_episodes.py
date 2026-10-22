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


# 第一段经历：A（第 0~11 拍）
A_snaps = []
for _ in range(12):
    ctx.tick(scene_obs(sp, 1), record=False)
    A_snaps.append(code())
# 第二段经历：B（第 12~23 拍）
B_snaps = []
for _ in range(12):
    ctx.tick(scene_obs(sp, 2), record=False)
    B_snaps.append(code())
b.binder.flush()

A_code = A_snaps[6]       # 第 6 拍属于 A
B_code = B_snaps[6]       # 第 18 拍属于 B
slotA = int(b.time_ring.ids[6])
slotB = int(b.time_ring.ids[18])

for tag, slot, own, other in (("A 的时刻", slotA, A_code, B_code),
                              ("B 的时刻", slotB, B_code, A_code)):
    got = cue(slot)
    print(f"点{tag}: 想到 {len(got):4d}  命中本段 {len(got & own) / max(1, len(own)):.2f}"
          f"  命中另一段 {len(got & other) / max(1, len(other)):.2f}"
          f"  精确 {len(got & own) / max(1, len(got)):.2f}")
