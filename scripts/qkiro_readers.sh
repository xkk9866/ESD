#!/bin/bash
# Reader generalization for ESD on the two other int4 readers (GPU 0). Existing prediction files are skipped by reader.py.
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=0 VLLM_UTIL=0.60
D4=nq,hotpotqa,musique,2wiki
M="{esd,esd_marg,ftce_m_esd,ftce_ok,adacomp,exitce,ecd,compact_qwen,full,provence,llm_summary,ftce_r4,longllmlingua}"
run kr_l8 $PY src/reader.py --name llama8b --model models/Llama-3.1-8B-Instruct --quant bitsandbytes --datasets $D4 --methods "$M"
run kr_q3 $PY src/reader.py --name qwen3b --model models/Qwen2.5-3B-Instruct-AWQ --datasets $D4 --methods "$M"
touch stages/kr_all.done; echo KR_DONE
