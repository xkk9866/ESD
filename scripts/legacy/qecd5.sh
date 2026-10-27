#!/bin/bash
# v2: sufficiency stopping. Re-decode every chain variant, then read with the three int4 readers (<14B), GPU 6 only.
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=6
run s_ecd    $PY src/ecd_run.py --ckpt ckpt/ecd  --ctx chain  --name ecd     --ratios 2,4,8,16
run s_ecdi0  $PY src/ecd_run.py --ckpt ckpt/ecd  --ctx none   --name ecdi0   --ratios 16 --taus ""
run s_ecdr   $PY src/ecd_run.py --ckpt ckpt/ecd  --ctx random --name ecdrand --ratios 16 --taus ""
run s_cnech  $PY src/ecd_run.py --ckpt ckpt/cne  --ctx chain  --name cnech   --ratios 16
run s_ftceb  $PY src/ecd_run.py --ckpt ckpt/ftce --ctx none   --name ftceb   --ratios ""
V="ecd,ecd_unit,ecd_s0.3,ecd_s0.7,ecd_s0.9,ecd_p2,ecd_p4,ecd_p8,ecd_p16,ecdi0,ecdi0_unit,ecdi0_p16,ecdrand,ecdrand_unit,ecdrand_p16,cnech,cnech_unit,cnech_s0.3,cnech_s0.7,cnech_s0.9,cnech_p16,ftceb,ftceb_unit,ftceb_s0.3,ftceb_s0.7,ftceb_s0.9"
run s_r7     $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --methods "{$V}"
run s_an7    $PY src/ecd_analysis.py --reader qwen7b --names ecd,ecd_unit,ecdi0,ecdrand,cnech,ftceb --out results/ecd_analysis.json
touch stages/v2_r7.done; echo V2_R7_DONE
RM="{full,closedbook,trunc_match,bm25_match,bge_match,ftce_match,recomp_ext_match,llmlingua2,provence,longllmlingua,llm_summary,ftce_p4,ftce_p16,ecd,ecd_unit,ecd_p4,ecd_p16,ecdi0,ecdrand,cnech,ftceb}"
run s_ll     $PY src/reader.py --name llama8b --model models/Llama-3.1-8B-Instruct --quant bitsandbytes --methods "$RM"
run s_q3     $PY src/reader.py --name qwen3b --model models/Qwen2.5-3B-Instruct-AWQ --methods "$RM"
run s_ev7    $PY src/evaluate.py --reader qwen7b
run s_evl    $PY src/evaluate.py --reader llama8b
run s_ev3    $PY src/evaluate.py --reader qwen3b
touch stages/v2_all.done; echo V2_ALL_DONE
