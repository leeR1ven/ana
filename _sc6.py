import subprocess

import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
for cmd in ("pgrep -fc night.sh; pgrep -fc p43_multi2; pgrep -fc python3",
            "cd /root/ana && tail -4 night.log; tail -2 night_20.log 2>/dev/null",
            "free -g | head -2"):
    _, o, e = c.exec_command(cmd, timeout=120)
    print(o.read().decode("utf-8", "replace").strip()[:600])
    print("----")
c.close()
