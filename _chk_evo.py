import json

import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
s = c.open_sftp()
_, o, e = c.exec_command("cd /root/ana && grep '代：技能' dog.log | tail -2; "
                         "pgrep -fc dog_evolve", timeout=60)
print(o.read().decode("utf-8", "replace").strip())
try:
    with s.open("/root/ana/dog_evolve_result.json") as f:
        d = json.load(f)
    print("skills:", len(d.get("skills", {})))
    print("has gait:", "gait" in (d.get("skills") or {}))
    kw = d.get("best_kw") or {}
    print("best_kw:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items()})
    with open("/root/ana/best_from_evo.json", "w") as g:
        json.dump({"kw": kw}, g)
except Exception as ex:
    print("结果未写:", type(ex).__name__)
s.close()
c.close()
