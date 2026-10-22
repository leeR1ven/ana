import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agi import build_spec, build, Cortex
from agi.encoding import Observation
from agi.world import _blob

sp = build_spec("full")
b = build(sp, verbose=False)
ctx = Cortex(b)
eng = b.engine
ids = np.array(sorted(np.concatenate(
    [l.ids for n, l in b.layers.items() if n.endswith(".L5")]).tolist()))
px = sp.px


def mk(kind):
    img = np.zeros((px, px, 3), np.float32) + 0.05
    if kind == "A":
        img = np.maximum(img, _blob(px, 0.3, 0.4, 0.2, (0.95, 0.2, 0.1)))
    else:                                  # 绿色横条：面积与红球相近，形状/颜色不同
        img = np.maximum(img, _blob(px, 0.7, 0.6, 0.2, (0.1, 0.9, 0.2)))   # 同尺寸、不同位置/颜色
    return Observation(vision=img.astype(np.float32),
                       audio=np.zeros((sp.audio_bands, sp.audio_frames), np.float32),
                       touch=np.full(sp.touch_ch, 0.5, np.float32),
                       proprio=np.full(sp.proprio_ch, 0.5, np.float32),
                       balance=np.full(sp.balance_ch, 0.5, np.float32))


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


A_snaps, B_snaps = [], []
for _ in range(12):
    ctx.tick(mk("A"), record=False)
    A_snaps.append(code())
for _ in range(12):
    ctx.tick(mk("B"), record=False)
    B_snaps.append(code())
b.binder.flush()

A_code, B_code = A_snaps[6], B_snaps[6]
jab = len(A_code & B_code) / max(1, len(A_code | B_code))
print(f"两段码本身的重合 {jab:.2f}  (A {len(A_code)}, B {len(B_code)})")
for tag, slot, own, other in (("A", int(b.time_ring.ids[6]), A_code, B_code),
                              ("B", int(b.time_ring.ids[18]), B_code, A_code)):
    got = cue(slot)
    print(f"点{tag}的时刻: 想到 {len(got):4d}  本段 {len(got & own) / max(1, len(own)):.2f}"
          f"  另一段 {len(got & other) / max(1, len(other)):.2f}"
          f"  精确 {len(got & own) / max(1, len(got)):.2f}")
