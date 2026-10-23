import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
for cmd in ("cd /root/ana && tail -8 walkfar.log",
            "pgrep -fc p49_walk_far; uptime",
            "cd /root/ana && ls -l walk_far.json 2>/dev/null | head -1"):
    _, o, e = c.exec_command(cmd, timeout=120)
    print(o.read().decode("utf-8", "replace").strip()[:800])
    print("----")
c.close()
