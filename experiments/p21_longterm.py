import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agi import build_spec, build, Cortex
from agi.consolidate import Consolidator, recall_readout
from agi.dentate import dentate_code
from experiments.p1_recall import scene_obs


def trial(seed: int, cycles: int = 8, cap: int = 512):
    sp = build_spec("full", dentate_factor=4, seed=7 + seed)
    b = build(sp, verbose=False)
    eng = b.engine
    dg = np.asarray(b.ids["dentate"])
    ctx = Cortex(b)
    co = Consolidator(eng, lr=1.0, cap=cap)

    def code():
        return set(dentate_code(eng, dg).tolist())

    def run(sc, n=12):
        eng.act[:] = 0.0
        eng.reset_adaptation()
        for _ in range(n):
            ctx.tick(scene_obs(sp, sc), record=False)
        return code()

    for _ in range(cycles):
        for sc in (1, 2):
            run(sc)
            co.step(list(code()))
    A = run(1)
    B = run(2)
    co.flush()

    eng.act[:] = 0.0
    eng.reset_adaptation()
    for _ in range(10):
        eng.clear_external()
        eng.set_input(np.zeros(b.encoder.n_in, np.float32))
        eng.set_external(list(A), np.ones(len(A), np.float32))
        eng.step()
    got = recall_readout(eng, dg)
    return (len(A), len(B), len(got & A) / max(1, len(A)),
            len(got & B) / max(1, len(B)),
            len((got & A) | (got & B)) / max(1, len(got)))


if __name__ == "__main__":
    for s in range(3):
        la, lb, ha, hb, prec = trial(s)
        print(f"seed {s}: |A| {la} |B| {lb} 命中A {ha:.2f} 命中B {hb:.2f} 精确 {prec:.2f}")
