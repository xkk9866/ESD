"""block until stages/<name>.done exists on the server: python scripts/remote/waitstage.py name"""
import sys, time, subprocess
while True:
    o = subprocess.run([sys.executable, "scripts/remote/rx.py", f"test -f ~/research/qrag/stages/{sys.argv[1]}.done && echo YES; grep -h FAIL ~/research/qrag/logs/qspa_C2.out"], capture_output=True, text=True).stdout or ""
    if "YES" in o or "FAIL" in o:
        print(o); break
    time.sleep(300)
