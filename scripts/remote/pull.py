"""Upload a local script to the server, run it there (cwd ~/research/qrag, PYTHONPATH=src, venv python), save stdout locally.
python scripts/remote/pull.py <local_script> <local_out> [args...]"""
import os  # public release: SSH configuration comes from the environment
import sys, warnings
warnings.filterwarnings("ignore")
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(os.environ["SSH_HOST"], port=int(os.environ.get("SSH_PORT", "22")), username=os.environ["SSH_USER"], password=os.environ.get("SSH_PASSWORD"), timeout=30)
s = c.open_sftp(); s.put(sys.argv[1], "/tmp/_kiro_job.py"); s.close()
_, o, e = c.exec_command("cd ~/research/qrag && PYTHONPATH=src .venv/bin/python /tmp/_kiro_job.py " + " ".join(sys.argv[3:]), timeout=1700)
open(sys.argv[2], "wb").write(o.read())
sys.stderr.write(e.read().decode("utf-8", "replace")[-3000:])
c.close()
