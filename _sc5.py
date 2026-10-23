import json

import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
for cmd in ("cd /root/ana && tail -3 night.log",
            "cd /root/ana && ls -t night_*.log | head -1 | xargs -I@ sh -c 'grep 代 @ | tail -1'"):
    _, o, e = c.exec_command(cmd, timeout=120)
    print(o.read().decode("utf-8", "replace").strip()[:300])
    print("----")
_, o, e = c.exec_command(
    "cd /root/ana && export OMP_NUM_THREADS=1 ANA_GO2_XML=/root/go2models/scene.xml && "
    "timeout 1700 python3 experiments/p47_morning.py 2>&1 | tail -3", timeout=2400)
print(o.read().decode("utf-8", "replace").strip()[-500:])
s = c.open_sftp()
try:
    with s.open("/root/ana/morning_report.json") as f:
        d = json.load(f)
    print("passed:", len(d["passed"]), d["passed"])
    with open(r"F:\一种AGI关联神经元架构\ana\morning_report.json", "w", encoding="utf-8") as g:
        json.dump(d, g, ensure_ascii=False, indent=2)
except Exception as ex:
    print("report:", type(ex).__name__)
s.close()
c.close()
