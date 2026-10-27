#!/bin/bash
# Lane A (GPU 5): PSD on the four benchmarks -> same-length FT-CE -> main reader
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=5 VLLM_UTIL=0.40
D4=nq,hotpotqa,musique,2wiki
run k_psd   $PY src/psd.py --ckpt ckpt/esd --name psd --datasets $D4
run k_mft   $PY src/baselines.py --method ftce --ckpt ckpt/ftce --modes m_psd --datasets $D4
run k_r7    $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --datasets $D4 --methods "{psd,psd_*,ftce_m_psd}"
touch stages/k5a_all.done; echo K5A_DONE
