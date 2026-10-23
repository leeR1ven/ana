p = "agi/build.py"
s = open(p, encoding="utf-8").read()
old = """                u = cpg[int((m // 3) * (_cpg / 4.0)) % _cpg]"""
old2 = """                # 按腿分配：同一条腿的三条肌肉共用同一个 CPG 相位
                u = cpg[(m // 3) % _cpg]"""
new = """                # 手工 trot 连线：对角腿同相（FL+RR 同相，FR+RL 反相），
                # 这是真实狗小跑的基本相位关系——先给一个"粗糙但能做"的解。
                _leg = m // 3
                if str(getattr(spec, "trot", "")).lower() in ("1", "true", "yes"):
                    _phase = (0.0, 0.5, 0.5, 0.0)[_leg % 4]
                    u = cpg[int(round(_phase * _cpg)) % _cpg]
                else:
                    u = cpg[(m // 3) % _cpg]"""
if old2 in s:
    s = s.replace(old2, new, 1)
elif old in s:
    s = s.replace(old, new, 1)
else:
    raise SystemExit("anchor not found")
open(p, "w", encoding="utf-8").write(s)

p = "agi/spec.py"
s = open(p, encoding="utf-8").read()
if "trot:" not in s:
    s = s.replace("    cpg_units: int = 0",
                  "    trot: bool = False                            # 手工 trot 相位（对角腿同相）\n"
                  "    cpg_units: int = 0")
    open(p, "w", encoding="utf-8").write(s)
print("trot wiring ok")
