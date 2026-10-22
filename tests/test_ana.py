# -*- coding: utf-8 -*-
"""不依赖 pytest 的自检：python tests/test_ana.py

分五组：编码 / 引擎 / 架构不变量 / 联想生长 / 闭环。
每组都只跑小规模（tiny），几秒钟内跑完。
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agi import PlasticConfig, PlasticRule, build, build_spec          # noqa: E402
from agi.build import G                                                 # noqa: E402
from agi.check import check                                             # noqa: E402
from agi.encoding import Observation, thermometer                        # noqa: E402
from agi.engine import Engine, EngineConfig                             # noqa: E402
from agi.loop import Cortex                                             # noqa: E402
from agi.world import VirtualDog, sample_observation                    # noqa: E402

TINY = None


def tiny():
    global TINY
    if TINY is None:
        TINY = build(build_spec("tiny"), verbose=False)
    return TINY


# ---------------------------------------------------------------- 编码
def test_thermometer_exactly_half():
    v = np.array([0.0, 0.05, 0.37, 0.5, 0.95, 1.0], np.float32)
    pos, neg = thermometer(v, 10)
    assert pos.shape == (6, 10) and neg.shape == (6, 10)
    assert np.all(pos.sum(1) + neg.sum(1) == 10)
    assert pos[0].sum() == 0 and neg[0].all()
    assert pos[-1].all() and neg[-1].sum() == 0


def test_encoder_one_hot_pairs():
    b = tiny()
    sp = b.spec
    obs = sample_observation(sp, np.random.default_rng(0), 0)
    buf = b.encoder.encode(obs)
    assert buf.shape == (b.encoder.n_in,)
    assert set(np.unique(buf)).issubset({0.0, 1.0})
    a, e = b.encoder.layout["vision"]
    arr = buf[a:e].reshape(-1, 2, sp.n_level)
    assert np.all(arr.sum(axis=1) == 1), "每档的正反一对必须恰好亮一个"
    # 每个模态的总活跃数恒定
    n_ch = sum((e - a) // (2 * sp.n_level) * sp.n_level for a, e in b.encoder.layout.values())
    assert int(buf.sum()) == n_ch


def test_encoder_black_is_a_reading():
    """全黑不是"没有信号"：反细胞全亮，活跃总数仍然恒定。"""
    b = tiny()
    sp = b.spec
    buf = b.encoder.encode(Observation(tick=0))
    n_ch = sum((e - a) // (2 * sp.n_level) * sp.n_level
               for a, e in b.encoder.layout.values())
    assert int(buf.sum()) == n_ch


# ---------------------------------------------------------------- 引擎
def test_engine_basic():
    eng = Engine(EngineConfig(target_active=0))
    a = eng.add_block(4, kind="input", thr=0.0, soft=1.0, counted=False)
    b = eng.add_block(2, kind="layer", thr=0.5, soft=0.5)
    eng.connect(np.array([b[0], b[0]]), np.array([a[0], a[1]]), np.array([0.5, 0.5]))
    eng.link(a[2:4], b[1:2].repeat(2))
    eng.finalize()
    eng.input_size = len(a)          # 感觉细胞：每拍直接等于输入
    eng.set_input(np.array([1.0, 1.0, 0.0, 0.0], np.float32))
    eng.step()      # 第一拍：感觉细胞就位
    eng.step()      # 第二拍：下游吃到这一拍的输入
    act = eng.act_np
    assert np.allclose(act[a], [1.0, 1.0, 0.0, 0.0])
    assert act[b[0]] > 0.2, "两个 0.5 输入应该过 0.5 的门槛"
    assert act[b[1]] == 0.0


def test_engine_topk_exact():
    b = tiny()
    eng = b.engine
    for _ in range(12):
        eng.set_input(b.encoder.encode(sample_observation(b.spec, np.random.default_rng(1), 0)))
        eng.step()
    for name, L in b.layers.items():
        n = int(eng.selected[L.ids].sum())
        assert n == L.k, f"{name}: 被选中 {n} != top-k {L.k}"


def test_engine_gain_control():
    eng = Engine(EngineConfig(target_active=1, gain_lr=0.05, gain_range=(-0.5, 0.5)))
    a = eng.add_block(20, kind="layer", thr=0.0, soft=1.0)
    eng.finalize()
    eng.set_input(np.zeros(0, np.float32)) if False else None
    for _ in range(40):
        eng.set_external(a, np.ones(20, np.float32))
        eng.step()
    assert eng.gain > 0.0, "活跃过多时门槛增益应上升"


# ---------------------------------------------------------------- 架构
def test_architecture_invariants():
    b = tiny()
    issues = check(b, verbose=False)
    assert not issues, issues
    eng = b.engine
    # 1:1 专用线：保留的"区"前额叶神经元恰好一条入线；被筛掉的 0 条
    region_pfc = np.concatenate(list(b.pfc_groups.values()))
    masks = [b.filtered.get(f"{r}.L{d}", np.zeros(len(g), bool))
             for (r, d), g in b.pfc_groups.items()]
    fdrop = np.concatenate(masks) if masks else np.zeros(len(region_pfc), bool)
    indeg = np.bincount(eng.dst, minlength=eng.n)
    assert np.all(indeg[region_pfc[~fdrop]] == 1), "1:1 下保留的前额叶必须一人一根线"
    assert np.all(indeg[region_pfc[fdrop]] == 0), "被筛掉的单元不该有入线"
    assert 0 < int(fdrop.sum()) < len(fdrop), "常亮筛选应该筛掉一部分、不是全部"
    # v0 没有任何联想边
    assert int((eng.group == G.PL_PFC).sum()) == 0
    assert int((eng.group == G.PLASTIC).sum()) == 0
    # 每层都有中继 + 前额叶组
    for key in b.layers:
        L = b.layers[key]
        assert (L.region, L.depth) in b.ids["relay"]
        assert (L.region, L.depth) in b.pfc_groups
    # 运动区：12 块肌肉 × 10 档
    assert len(b.motor["force"]) == b.spec.n_muscle
    for f in b.motor["force"].values():
        assert len(f) == b.spec.n_force
        t = eng.thr[f]
        assert np.all(np.diff(t) >= 0), "力度档门槛必须单调"


def test_filter_removes_invariant():
    b = tiny()
    assert b.filtered, "默认应该做常亮筛选"
    total = sum(int(m.sum()) for m in b.filtered.values())
    assert total > 0
    # 被筛掉的神经元不能还有向前的专用线
    eng = b.engine
    for name, L in b.layers.items():
        bad = b.filtered[name]
        if bad.any():
            assert not (np.isin(eng.src, L.ids[bad]) & (eng.group == G.PFC)).any()
    # 关掉筛选就会多出边
    b2 = build(build_spec("tiny", pfc_filter=False), verbose=False)
    assert b2.engine.n_edges > eng.n_edges


def test_architecture_two_modes():
    """fan_in=6 的组合检测器也要能建、能过自检（留着以后对照）。"""
    sp = build_spec("tiny", pfc_fan_in=6)
    b = build(sp, verbose=False)
    assert not check(b)
    region_pfc = np.concatenate(list(b.pfc_groups.values()))
    masks = [b.filtered.get(f"{r}.L{d}", np.zeros(len(g), bool))
             for (r, d), g in b.pfc_groups.items()]
    fdrop = np.concatenate(masks) if masks else np.zeros(len(region_pfc), bool)
    indeg = np.bincount(b.engine.dst, minlength=b.engine.n)
    assert np.all(indeg[region_pfc[~fdrop]] == 6)


def test_behavior_loop():
    b = tiny()
    ctx = Cortex(b, trace=False)
    dog = VirtualDog(b.spec, seed=1)
    obs = dog.observe()
    seen = 0
    for _ in range(20):
        tk = ctx.tick(obs, record=False)
        seen = max(seen, tk.pfc_active)
        assert np.all(np.isfinite(tk.forces))
        assert np.all((tk.forces >= 0) & (tk.forces <= 1))
        dog.step(tk.forces)
        obs = dog.observe()
    assert seen > 0


def test_trace_save(tmp_path="ana_trace_test.npz"):
    b = tiny()
    ctx = Cortex(b, trace=True)
    ctx.run(12)
    a = ctx.trace.arrays()
    assert a["forces"].shape[0] == 12
    assert np.isfinite(a["forces"]).all()
    assert a["pfc_active"].mean() > 0
    ctx.trace.save(tmp_path)
    assert os.path.exists(tmp_path)
    os.remove(tmp_path)
    p = tmp_path.replace(".npz", "_pfc.npz")
    if os.path.exists(p):
        os.remove(p)


# ---------------------------------------------------------------- 联想生长
def test_plastic_temporal_order():
    """A 先亮、B 后亮 → 只长 A→B，不长 B→A。"""
    eng = Engine(EngineConfig(target_active=0))
    ids = eng.add_block(6, kind="pfc", region="pfc", thr=0.0, soft=1.0)
    eng.finalize()
    rule = PlasticRule(PlasticConfig(enabled=True, eta=0.2, topk_out=8, min_w=1e-4))
    A, B = ids[:2], ids[2:4]
    rest = np.zeros(0, np.int64)
    for _ in range(6):
        eng.act[:] = 0.0
        eng.act[A] = 1.0
        rule.apply(eng, A)
        eng.act[:] = 0.0
        eng.act[B] = 1.0
        rule.apply(eng, B)
        eng.act[:] = 0.0
        rule.apply(eng, rest)        # 静息一拍：避免下一轮 A 变成"跟在 B 后面"
    rule.flush(eng)          # 边是攒着批量写进去的
    m = eng.group == G.PL_PFC
    assert int(m.sum()) > 0, "顺序共现应该长出边"
    pairs = {(int(s), int(d)) for s, d in zip(eng.src[m], eng.dst[m])}
    assert (A[0], B[0]) in pairs or (int(A[0]), int(B[0])) in pairs
    assert all((b, a) not in pairs for a in A for b in B), "反向不该等强成边"
    assert np.all(eng.w[m] > 0)


def test_adaptation_habituates():
    eng = Engine(EngineConfig(target_active=0))
    a = eng.add_block(2, kind="pfc", thr=0.0, soft=1.0)
    eng.finalize()
    eng.set_adaptation(a, 0.5)
    eng.set_external(a, np.array([1.0, 0.0], np.float32))
    acts = []
    for _ in range(8):
        eng.step()
        acts.append(float(eng.act_np[a[0]]))
    assert acts[0] > acts[-1], f"持续输入下响应应该衰减：{acts}"
    # 换一个输入：另一个单元应当有响应（不是整体压死）
    eng.set_external(a, np.array([0.0, 1.0], np.float32))
    eng.step()
    assert float(eng.act_np[a[1]]) > 0.1


def test_pmi_gate():
    """特异性闸门：同样的共现次数，稀有搭档的闸门要高于万金油搭档。"""
    eng = Engine(EngineConfig(target_active=0))
    ids = eng.add_block(6, kind="pfc", region="p", thr=0.0, soft=1.0)
    eng.finalize()
    rule = PlasticRule(PlasticConfig(enabled=True, topk_out=100,
                                     pmi_gate=True, pmi_steep=1.2, pmi_floor=0.12))
    # 稀有对 0→1（各自只出现 10 次，一起出现 10 次）
    # 万金油 4→5（各自出现 80 次，也只一起出现 10 次）
    rule._cnt_unit = np.array([10, 10, 0, 0, 80, 80], np.float32)
    rule.t = 100
    eng.add_edges(np.array([0, 4], np.int32), np.array([1, 5], np.int32),
                  np.array([0.5, 0.5], np.float32), group=40)
    rule._cnt_edge = np.array([10.0, 10.0], np.float32)   # 与边对齐的共现计数
    g = rule.gates(eng)
    assert g[0] > g[1], f"稀有搭档应更被保留：{g}"
    assert g[0] > 0.5 and g[1] < 0.5


def test_consolidation_freezes():
    """用得多就变难改：冻结之后同样的反复使用不能再把权重推上去。"""
    def run(consolidate):
        eng = Engine(EngineConfig(target_active=0))
        ids = eng.add_block(2, kind="pfc", region="p", thr=0.0, soft=1.0)
        eng.finalize()
        rule = PlasticRule(PlasticConfig(enabled=True, eta=0.3, topk_out=8,
                                         full_cross=100, min_change=0.0,
                                         flush_every=1, flush_edges=1000,
                                         consolidate=consolidate,
                                         use_ref=2.0, freeze_at=5.0))
        for _ in range(40):
            rule.reinforce(eng, np.array([ids[0]]), np.array([ids[1]]))
            rule.flush(eng)
        m = eng.group == 40
        return float(eng.w[m][0]), bool(rule._frozen[0]) if len(rule._frozen) else False

    w_cons, frozen = run(True)
    w_plain, _ = run(False)
    assert w_cons < w_plain - 0.1, f"固化后应该涨得更少：{w_cons:.2f} vs {w_plain:.2f}"
    assert frozen


def test_plastic_disabled_is_noop():
    b = tiny()
    rule = PlasticRule(PlasticConfig(enabled=False))
    out = rule.apply(b.engine, b.pfc_code())
    assert out["edges"] == 0 and out["grown"] == 0
    assert int((b.engine.group == G.PL_PFC).sum()) == 0


def test_plastic_topk():
    eng = Engine(EngineConfig(target_active=0))
    ids = eng.add_block(10, kind="pfc", region="pfc", thr=0.0, soft=1.0)
    eng.finalize()
    rule = PlasticRule(PlasticConfig(enabled=True, eta=0.3, topk_out=2, min_w=1e-6))
    for _ in range(6):
        for a in [ids[0]]:
            eng.act[:] = 0.0
            eng.act[a] = 1.0
            rule.apply(eng, np.array([a]))
            eng.act[:] = 0.0
            eng.act[ids[1:]] = 1.0
            rule.apply(eng, ids[1:])
    rule.flush(eng)
    m = eng.group == G.PL_PFC
    src = eng.src[m]
    if len(src):
        _, cnt = np.unique(src, return_counts=True)
        assert cnt.max() <= 2, f"每个单元最多 2 条出边，实际 {cnt.max()}"


# ---------------------------------------------------------------- runner
TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    ok = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
            ok += 1
        except Exception:
            print(f"  FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{ok}/{len(TESTS)} 通过")
    return 0 if ok == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
