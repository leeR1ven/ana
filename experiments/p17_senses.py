import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agi import build_spec, build, Cortex
from agi.encoding import Observation

sp = build_spec("full")
b = build(sp, verbose=False)
eng = b.engine


def base_obs(vision_seed=1):
    r = np.random.default_rng(vision_seed)
    return Observation(
        vision=r.random((sp.px, sp.px, 3), dtype=np.float32),
        audio=np.zeros((sp.audio_bands, sp.audio_frames), np.float32),
        touch=np.full(sp.touch_ch, 0.5, np.float32),
        proprio=np.full(sp.proprio_ch, 0.5, np.float32),
        balance=np.full(sp.balance_ch, 0.5, np.float32))


def run(obs, n=16):
    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(n):
        eng.clear_external()
        eng.set_input(b.encoder.encode(obs))
        eng.step()


def code(region):
    L = b.layers[f"{region}.L5"]
    return set(np.flatnonzero(np.asarray(eng.act)[L.ids] > 0.5).tolist())


def vary(modality):
    a = base_obs()
    c = base_obs()
    if modality == "audio":
        low = np.zeros((sp.audio_bands, sp.audio_frames), np.float32)
        low[: sp.audio_bands // 4] = 0.9
        high = np.zeros((sp.audio_bands, sp.audio_frames), np.float32)
        high[3 * sp.audio_bands // 4:] = 0.9
        a.audio, c.audio = low, high
    elif modality == "touch":
        a.touch = np.full(sp.touch_ch, 0.1, np.float32)
        c.touch = np.full(sp.touch_ch, 0.9, np.float32)
    elif modality == "proprio":
        a.proprio = np.full(sp.proprio_ch, 0.2, np.float32)
        c.proprio = np.full(sp.proprio_ch, 0.8, np.float32)
    elif modality == "balance":
        a.balance = np.full(sp.balance_ch, 0.2, np.float32)
        c.balance = np.full(sp.balance_ch, 0.8, np.float32)
    return a, c


for m in ("audio", "touch", "proprio", "balance"):
    a, c = vary(m)
    run(a); A = code(m); run(a); A2 = code(m); run(c); B = code(m)
    same = (A == A2)
    j = len(A & B) / max(1, len(A | B))
    print(f"{m:8s} size {len(A):4d}  same-image-consistent {same}  overlap(A,B) {j:.2f}")
