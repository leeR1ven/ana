import json

import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
for cmd in ("cd /root/ana && tail -12 walkfar.log",
            "pgrep -fc p49_walk_far"):
    _, o, e = c.exec_command(cmd, timeout=120)
    print(o.read().decode("utf-8", "replace").strip()[:900])
    print("----")
s = c.open_sftp()
try:
    with s.open("/root/ana/walk_far.json") as f:
        d = json.load(f)
    with open(r"F:\一种AGI关联神经元架构\ana\walk_far.json", "w", encoding="utf-8") as g:
        json.dump({"kw": d["kw"]}, g, ensure_ascii=False, indent=2)
    print("纪录 dx", round(d["dx"], 3), "倒下", d["fell"])
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d["kw"].items()})
except Exception as ex:
    print("walk_far:", type(ex).__name__)
s.close()
c.close()
