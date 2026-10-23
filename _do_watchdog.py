import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
s = c.open_sftp()
s.put("_watchdog.sh", "/root/watchdog.sh")
s.close()
cmd = ("sed -i '1s/^\\xEF\\xBB\\xBF//' /root/watchdog.sh; sed -i 's/\\r$//' /root/watchdog.sh; "
       "chmod +x /root/watchdog.sh; "
       "setsid nohup bash /root/watchdog.sh > /dev/null 2>&1 < /dev/null & "
       "sleep 20; pgrep -fc watchdog.sh; pgrep -fc p49_walk_far")
_, o, e = c.exec_command(cmd, timeout=600)
print(o.read().decode("utf-8", "replace").strip())
print(e.read().decode("utf-8", "replace").strip()[-200:])
c.close()
