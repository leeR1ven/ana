import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
s = c.open_sftp()
s.put("_wf4.sh", "/root/wf4.sh")
s.close()
cmd = ("sed -i '1s/^\\xEF\\xBB\\xBF//' /root/wf4.sh; "
       "sed -i 's/\\r$//' /root/wf4.sh; bash /root/wf4.sh")
_, o, e = c.exec_command(cmd, timeout=2400)
print(o.read().decode("utf-8", "replace").strip()[-700:])
print(e.read().decode("utf-8", "replace").strip()[-200:])
c.close()
