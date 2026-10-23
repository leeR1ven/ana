import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
s = c.open_sftp()
s.put("experiments/p49_walk_far.py", "/root/ana/experiments/p49_walk_far.py")
s.put("_wf.sh", "/root/wf.sh")
s.put("champion.json", "/root/ana/champion.json")
s.close()
_, o, e = c.exec_command("sed -i 's/\\r$//' /root/wf.sh; bash /root/wf.sh", timeout=1800)
print(o.read().decode("utf-8", "replace").strip()[-900:])
print(e.read().decode("utf-8", "replace").strip()[-200:])
c.close()
