p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()
# 所有候选都开 trot 相位（手工粗糙初解），并保证 cpg_units 至少 4
s = s.replace('    out["balance_sectors"] = 36\n    out["balance_per_sector"] = 20',
              '    out["trot"] = True\n    out["balance_sectors"] = 36\n    out["balance_per_sector"] = 20')
s = s.replace('    out["cpg_units"] = int(np.clip(kw.get("cpg_units", 8) + rng.integers(-2, 3) * scale, 0, 16))',
              '    out["cpg_units"] = int(np.clip(kw.get("cpg_units", 8) + rng.integers(-2, 3) * scale, 4, 16))')
s = s.replace('        balance_sectors=36, balance_per_sector=20,',
              '        trot=True,\n        balance_sectors=36, balance_per_sector=20,')
s = s.replace('        cpg_units=int(rng.choice([0, 2, 4, 8, 12, 16])),',
              '        cpg_units=int(rng.choice([4, 6, 8, 12, 16])),')
s = s.replace('    seed.update(balance_sectors=36, balance_per_sector=20,',
              '    seed.update(trot=True, balance_sectors=36, balance_per_sector=20,')
open(p, "w", encoding="utf-8").write(s)
print("trot in all candidates")
