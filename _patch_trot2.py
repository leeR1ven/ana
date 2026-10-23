p = "agi/build.py"
s = open(p, encoding="utf-8").read()
old = "                u = cpg[m % _cpg]"
new = ("                # 手工 trot 连线：对角腿同相（FL+RR 同相，FR+RL 反相）\n"
       "                _leg = m // 3\n"
       "                if str(getattr(spec, \"trot\", \"\")).lower() in (\"1\", \"true\", \"yes\"):\n"
       "                    _phase = (0.0, 0.5, 0.5, 0.0)[_leg % 4]\n"
       "                    u = cpg[int(round(_phase * _cpg)) % _cpg]\n"
       "                else:\n"
       "                    u = cpg[m % _cpg]")
assert old in s, "anchor missing"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)

p2 = "agi/spec.py"
t = open(p2, encoding="utf-8").read()
if "trot:" not in t:
    t = t.replace("    cpg_units: int = 0",
                  "    trot: bool = False                            # 手工 trot 相位（对角腿同相）\n"
                  "    cpg_units: int = 0")
    open(p2, "w", encoding="utf-8").write(t)
print("trot patch ok")
