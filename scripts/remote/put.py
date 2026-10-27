"""upload local files to ~/research/qrag/<same relative path>: python scripts/remote/put.py src/a.py scripts/b.sh"""
import os  # public release: SSH configuration comes from the environment
import sys, warnings
warnings.filterwarnings("ignore")
import paramiko
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(os.environ["SSH_HOST"], port=int(os.environ.get("SSH_PORT", "22")), username=os.environ["SSH_USER"], password=os.environ.get("SSH_PASSWORD"), timeout=30)
s = c.open_sftp()
for f in sys.argv[1:]:
    dst = "/home/yaokang/research/qrag/" + (f.split("scripts/", 1)[1] if f.startswith("scripts/") else f)
    s.put(f, dst); print("put", dst)
s.close(); c.close()
