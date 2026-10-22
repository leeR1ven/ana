import os
import sys

import paramiko

HOST, PORT, USER = "i-1.gpushare.com", 21409, "root"
PASS = os.environ.get("ANA_SSH_PASS", "")
local = sys.argv[1]
remote = sys.argv[2]
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
sftp = cli.open_sftp()
sftp.put(local, remote)
print("uploaded", local, "->", remote, os.path.getsize(local), "bytes")
sftp.close()
cli.close()
