import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
s = c.open_sftp()
s.put("agi/dog_tasks.py", "/root/ana/agi/dog_tasks.py")
s.put("agi/build.py", "/root/ana/agi/build.py")
s.put("agi/spec.py", "/root/ana/agi/spec.py")
s.put("experiments/p49_walk_far.py", "/root/ana/experiments/p49_walk_far.py")
s.close()
cmd = ("pkill -f p49_walk_far || true; sleep 2; bash /root/wf5.sh")
_, o, e = c.exec_command(cmd, timeout=1800)
print(o.read().decode("utf-8", "replace").strip()[-800:])
print(e.read().decode("utf-8", "replace").strip()[-200:])
c.close()
