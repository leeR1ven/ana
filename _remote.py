import os
import sys

import paramiko

HOST = "i-1.gpushare.com"
PORT = 21409
USER = "root"
PASS = os.environ.get("ANA_SSH_PASS", "")

cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "echo connected"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
stdin, stdout, stderr = cli.exec_command(cmd, timeout=600)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
print(out.strip())
if err.strip():
    print("[stderr]", err.strip()[:800])
cli.close()
