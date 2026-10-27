#!/bin/bash
# Lane B (GPU 5): PSD on open retrieval, round-0 policy, pool sizes -> readers for them
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=5 VLLM_UTIL=0.40
O=hotpotqa_open,musique_open; P=musique_k5,musique_k10
run k_psdo  $PY src/psd.py --ckpt ckpt/esd    --name psd  --datasets $O
run k_mfto  $PY src/baselines.py --method ftce --ckpt ckpt/ftce --modes m_psd --datasets $O
run k_psd0  $PY src/psd.py --ckpt ckpt/esd_r0 --name psd0 --datasets hotpotqa,musique --levels ""
run k_psdp  $PY src/psd.py --ckpt ckpt/esd    --name psd  --datasets $P --levels ""
run k_r7o   $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --datasets $O --max_model_len 16384 --methods "{full,esd,esd_marg,psd,psd_*,ftce_m_esd,ftce_m_psd,ftce_r4,bge_m_esd,adacomp,exitce,provence,llmlingua2,ecd}"
run k_r7b   $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --datasets hotpotqa,musique,$P --methods "{psd0,psd0_*,psd,esd,ecd,ftceb}"
touch stages/k5b_all.done; echo K5B_DONE
