p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()
s = s.replace('    out["trot"] = True', '    out["trot"] = True\n    out["balance_to_force"] = True')
s = s.replace('        trot=True,\n', '        trot=True,\n        balance_to_force=True,\n')
s = s.replace('    seed.update(trot=True,', '    seed.update(balance_to_force=True, trot=True,')
open(p, "w", encoding="utf-8").write(s)
print("balance_to_force in candidates")
