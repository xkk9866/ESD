#!/bin/bash
cd ~/research/qrag
kill 96117 2>/dev/null; sleep 3
sed -i 's/\r$//' qk5*.sh; chmod +x qk5*.sh
for q in qk5 qk5b qk5c; do setsid nohup ./$q.sh > logs/$q.log 2>&1 < /dev/null & done
echo launched
