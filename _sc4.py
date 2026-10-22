import json

import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
cmds = [
    "cd /root/ana && tail -3 night.log",
    "cd /root/ana && ls -t night_*.log | head -1 | xargs -I@ sh -c 'grep 代 @ | tail -1'",
    "ps -eo pcpu,args --sort=-pcpu | grep p43_multi2 | head -1",
]
for cmd in cmds:
    _, o, e = c.exec_command(cmd, timeout=120)
    print(o.read().decode("utf-8", "replace").strip()[:350])
    print("----")
s = c.open_sftp()
try:
    with s.open("/root/ana/champion.json") as f:
        d = json.load(f)
    print("champion", len(d.get("passes", [])), d.get("source"))
except Exception as ex:
    print("champion:", type(ex).__name__)
s.close()
c.close()
