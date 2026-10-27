"""Run a command on the compute server.
python scripts/remote/rx.py "<cmd>"            run a shell command
python scripts/remote/rx.py --py local.py [args] run a local python file remotely (cwd ~/research/qrag, read-only use)
python scripts/remote/rx.py --get remote local   download a file"""
import os  # public release: SSH configuration comes from the environment
import sys, warnings
warnings.filterwarnings("ignore")
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(os.environ["SSH_HOST"], port=int(os.environ.get("SSH_PORT", "22")), username=os.environ["SSH_USER"], password=os.environ.get("SSH_PASSWORD"), timeout=30)
if sys.argv[1] == "--get":
    s = c.open_sftp(); s.get(sys.argv[2], sys.argv[3]); s.close(); print("ok")
else:
    if sys.argv[1] == "--py":
        s = c.open_sftp(); s.put(sys.argv[2], "/tmp/_kiro_rx.py"); s.close()
        cmd = "cd ~/research/qrag && ~/miniconda3/bin/python /tmp/_kiro_rx.py " + " ".join(sys.argv[3:])
    else:
        cmd = sys.argv[1]
    _, o, e = c.exec_command(cmd, timeout=1200)
    sys.stdout.buffer.write(o.read()); sys.stdout.flush()
    sys.stderr.write(e.read().decode("utf-8", "replace"))
c.close()
