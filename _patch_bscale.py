p = "agi/dog_tasks.py"
s = open(p, encoding="utf-8").read()
old = '''    mag = float(np.hypot(roll, pitch))
    if mag < 1e-4:
        return
    theta = float(np.arctan2(roll, pitch))'''
new = '''    mag = float(np.hypot(roll, pitch)) * float(scale)
    if mag < 1e-4:
        return
    theta = float(np.arctan2(roll, pitch))'''
assert old in s, "anchor1"
s = s.replace(old, new, 1)
old2 = "def _drive_balance(eng, brain, roll, pitch, sectors=36, per=20):"
new2 = ("def _drive_balance(eng, brain, roll, pitch, sectors=36, per=20, scale=10.0):\n"
        "    # scale：倾角是弧度（0.05~0.9），而神经元阈值是 0.05~0.95。\n"
        "    # 不放大就永远够不到阈值——实测环从不点亮（ext 最大 0.03）。\n")
assert old2 in s, "anchor2"
s = s.replace(old2, new2, 1)
open(p, "w", encoding="utf-8").write(s)
print("balance scale added")
