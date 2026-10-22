import sys
sys.path.insert(0, "ana")
import numpy as np
from agi import build_spec, build, Cortex
from agi.hippocampus import HippoBinder
from experiments.p1_recall import scene_obs

sp = build_spec("tiny")
b = build(sp, verbose=False)
ctx = Cortex(b)
binder = HippoBinder(b.engine, b.pfc_ids, b.time_ring, lr=0.3, cap=512)

codes = []
for i in range(14):
    ctx.tick(scene_obs(sp, 1), record=False)
    code = set(b.pfc_code().tolist())
    codes.append(code)
    binder.step(list(code))
n = binder.flush()
print("绑定边", n, " 码大小", [len(c) for c in codes[::4]])

slot = b.time_ring.ids[6]          # 第 6 拍那格时间神经元
target = codes[6]
eng = b.engine
eng.act[:] = 0.0
eng.reset_adaptation()
for _ in range(10):
    eng.clear_external()
    eng.set_input(np.zeros(b.encoder.n_in, np.float32))
    eng.set_external(np.array([slot]), np.ones(1, np.float32))
    eng.step()
got = set(b.pfc_code().tolist())
rng = np.random.default_rng(1)
rand = set(rng.choice(b.pfc_ids, max(1, len(got)), replace=False).tolist())
print(f"点第6拍时间格: 想到 {len(got)} 个, 命中当时码 {len(got & target)}/{len(target)} "
      f"({len(got & target) / max(1, len(target)):.2f})  随机 {len(rand & target) / max(1, len(target)):.2f}")
