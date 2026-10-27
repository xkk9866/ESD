#!/bin/bash
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=6
while pgrep -f "name ecdrand" >/dev/null; do sleep 30; done
run d_ecd   $PY src/ecd_run.py --ckpt ckpt/ecd --ctx chain --name ecd --ratios 2,4,8,16
run d_r7    $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --methods "{ecd*,ftceb,cnech*}"
run e_an7   $PY src/ecd_analysis.py --reader qwen7b --out results/ecd_analysis.json
touch stages/ecd_r7.done; echo ECD_R7_DONE
RM="{full,closedbook,trunc_match,bm25_match,bge_match,ftce_match,recomp_ext_match,llmlingua2,provence,longllmlingua,llm_summary,ftce_p4,ftce_p8,ftce_p16,ftceb,ecd,ecd_p4,ecd_p8,ecd_p16,ecdi0,ecdi0_p16,cnech}"
run e_ll    $PY src/reader.py --name llama8b --model models/Llama-3.1-8B-Instruct --quant bitsandbytes --methods "$RM"
while pgrep -f "snapshot_download" >/dev/null; do sleep 30; done
run e_q3    $PY src/reader.py --name qwen3b --model models/Qwen2.5-3B-Instruct-AWQ --methods "$RM"
run e_ev7   $PY src/evaluate.py --reader qwen7b
run e_evl   $PY src/evaluate.py --reader llama8b
run e_ev3   $PY src/evaluate.py --reader qwen3b
touch stages/ecd_all.done; echo ECD_ALL_DONE
