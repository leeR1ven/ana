p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()
old = """        if best_any[1] > record:
            record = best_any[1]
            with open(os.path.join(ROOT, "walk_far.json"), "w", encoding="utf-8") as f:
                json.dump({"dx": best_any[1], "fell": best_any[2], "kw": best_any[0]},
                          f, ensure_ascii=False, indent=2)"""
new = """        # 分开存：会倒的最快 / 不摔的最快。之前只存"最快"，
        # 结果记录被会倒的那只覆盖，"不摔"的成果丢了。
        if best_any[1] > record:
            record = best_any[1]
            with open(os.path.join(ROOT, "walk_far.json"), "w", encoding="utf-8") as f:
                json.dump({"dx": best_any[1], "fell": best_any[2], "kw": best_any[0]},
                          f, ensure_ascii=False, indent=2)
        if (not best_nf[2]) and best_nf[1] > record_nf:
            record_nf = best_nf[1]
            with open(os.path.join(ROOT, "walk_far_nofall.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"dx": best_nf[1], "fell": False, "kw": best_nf[0]},
                          f, ensure_ascii=False, indent=2)"""
assert old in s, "anchor missing"
s = s.replace(old, new, 1)
s = s.replace("    record = 0.0\n", "    record = 0.0\n    record_nf = 0.0\n")
open(p, "w", encoding="utf-8").write(s)
print("separate no-fall archive")
