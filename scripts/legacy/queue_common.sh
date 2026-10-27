cd ~/research/qrag
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONPATH=src TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
mkdir -p logs stages
run() { local s=$1; shift
  [ -f stages/$s.done ] && { echo "skip $s"; return 0; }
  echo "=== $(date) START $s"; "$@" > logs/$s.log 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then touch stages/$s.done; echo "=== $(date) DONE $s"; else echo "=== $(date) FAIL $s rc=$rc"; fi; return $rc; }
waitfor() { while [ ! -f stages/$1.done ]; do sleep 60; done; }
MAIN='{full,closedbook,vericomp,trunc_match,bm25_match,bge_match,ftce_match,recomp_ext_match,llmlingua2,provence,llm_summary,longllmlingua}'
