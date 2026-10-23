p = "agi/build.py"
s = open(p, encoding="utf-8").read()
old = """        for i in range(_sectors):
            sel = bring[i * _per:(i + 1) * _per]
            half = _per // 2
            tgt = mpfc[(i * len(mpfc)) // _sectors]"""
new = """        # 接哪里：mpfc（绕远路，延迟大）或 force（脊髓级，快一个数量级）
        _b2f = str(getattr(spec, "balance_to_force", "")).lower() in ("1", "true", "yes")
        for i in range(_sectors):
            sel = bring[i * _per:(i + 1) * _per]
            half = _per // 2
            if _b2f:
                # 每个方向对应一条腿（36/4 = 9 个方向一条腿），直接接该腿的力度细胞
                _leg = int(i * spec.n_muscle / _sectors) // 3
                tgt = int(mforce[min(spec.n_muscle - 1, _leg * 3)][0])
            else:
                tgt = mpfc[(i * len(mpfc)) // _sectors]"""
assert old in s, "anchor missing"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)

p2 = "agi/spec.py"
t = open(p2, encoding="utf-8").read()
if "balance_to_force" not in t:
    t = t.replace("    balance_gain: float = 0.5",
                  "    balance_to_force: bool = False                 # 平衡环直连力度细胞（脊髓级）\n"
                  "    balance_gain: float = 0.5")
    open(p2, "w", encoding="utf-8").write(t)
print("balance_to_force ok")
