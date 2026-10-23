p = "agi/build.py"
s = open(p, encoding="utf-8").read()
old = """        for m in range(spec.n_muscle):
            u = cpg[m % _cpg]
            for f_cell in mforce[m]:"""
new = """        for m in range(spec.n_muscle):
            # 手工 trot 连线：对角腿同相（FL+RR 同相，FR+RL 反相）。
            # 先给一个"粗糙但能做"的解，再在此基础上随机变异。
            if str(getattr(spec, "trot", "")).lower() in ("1", "true", "yes"):
                _phase = (0.0, 0.5, 0.5, 0.0)[(m // 3) % 4]
                u = cpg[int(round(_phase * _cpg)) % _cpg]
            else:
                u = cpg[m % _cpg]
            for f_cell in mforce[m]:"""
assert old in s, "anchor missing"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("trot wiring patch ok")
