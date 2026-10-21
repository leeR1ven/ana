# -*- coding: utf-8 -*-
"""诊断：修好地基后，学到的联想边到底有没有"话语权"。"""
import sys
sys.path.insert(0, "ana")
import numpy as np
from agi import build_spec, build, Cortex, PlasticConfig, PlasticRule
from agi.build import G
from experiments.p1_memory import stim, run_sequence

sp = build_spec("xl", pfc_topk=0.02, pfc_adapt=0.01)
b = build(sp, verbose=False)
rule = PlasticRule(PlasticConfig(enabled=True, eta=0.10, eta_sim=0.02, topk_out=8,
                                 max_pairs=2048, decay_every=1000, decay=0.995,
                                 flush_every=100, flush_edges=20000), seed=3)
ctx = Cortex(b, plastic=rule)
cyc = ["left", "mid", "right", "mid"]
for t in range(600):
    ctx.tick(stim(cyc[(t // 8) % 4], sp), record=False)
rule.flush(b.engine)
eng = b.engine
m = eng.group == G.PL_PFC
print("联想边", int(m.sum()), " 平均权重", round(float(eng.w[m].mean()), 3) if m.any() else 0,
      " 平均传播权重", round(float(eng.wp[m].mean()), 4) if m.any() else 0)
pfc = b.pfc_ids
a = eng.act_np
d_all = np.bincount(eng.dst, weights=eng.wp * eng.sgn * a[eng.src], minlength=eng.n)
d_pl = np.bincount(eng.dst[m], weights=(eng.wp * eng.sgn * a[eng.src])[m], minlength=eng.n)
print("联想占前额叶驱动", round(float(d_pl[pfc].sum() / max(1e-9, d_all[pfc].sum())), 3),
      " 收到联想入边的前额叶单元占比",
      round(float((np.bincount(eng.dst[m], minlength=eng.n)[pfc] > 0).mean()), 3))

# 直接给 A 的码：撤掉感觉输入，只用外部驱动点亮 A，看 B 的单元会不会亮
ctx2 = Cortex(b, plastic=None)
code_a = run_sequence(ctx2, ["left"], sp, 12)[-1]
code_b = run_sequence(ctx2, ["mid"], sp, 12)[-1]
eng.act[:] = 0.0
eng.reset_adaptation()
for t in range(10):
    eng.clear_external()
    eng.set_input(np.zeros(b.encoder.n_in, np.float32))
    eng.set_external(code_a, np.ones(len(code_a), np.float32))
    eng.step()
got = b.pfc_code()
B = set(code_b.tolist())
print(f"只点 A（{len(code_a)} 个单元）之后：前额叶亮 {len(got)} 个，"
      f"其中 B 的单元 {len(B & set(got.tolist()))}/{len(B)}"
      f"（随机基线约 {len(got) * len(B) / max(1, len(pfc)):.1f}）")
