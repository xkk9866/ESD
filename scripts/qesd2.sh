#!/bin/bash
# ESD experiments after training. GPU 6 only; readers are int4 and < 14B.
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=6
waitfor esd_train
while pgrep -f "qecd5.sh" >/dev/null; do sleep 30; done
run x_esd     $PY src/esd_run.py --ckpt ckpt/esd    --name esd
run x_esdr0   $PY src/esd_run.py --ckpt ckpt/esd_r0 --name esdr0
run x_esdrand $PY src/esd_run.py --ckpt ckpt/esd    --name esdrand --ctx random
run x_mftce   $PY src/baselines.py --method ftce --ckpt ckpt/ftce --modes m_esd,ok
run x_mbge    $PY src/baselines.py --method bge --modes m_esd
run x_mbm25   $PY src/baselines.py --method bm25 --modes m_esd
run x_mrec    $PY src/baselines.py --method recomp_ext --modes m_esd
run x_mtr     $PY src/baselines.py --method trunc --modes m_esd
P=musique_k5,musique_k10,nq_k5,nq_k10
run x_pesd    $PY src/esd_run.py --ckpt ckpt/esd --name esd --datasets $P
run x_pecd    $PY src/ecd_run.py --ckpt ckpt/ecd --ctx chain --name ecd --datasets $P --ratios "" --taus ""
run x_pftce   $PY src/ecd_run.py --ckpt ckpt/ftce --ctx none --name ftceb --datasets $P --ratios "" --taus ""
NEW="esd,esd_marg,esdr0,esdr0_marg,esdrand,ftce_m_esd,ftce_ok,bge_m_esd,bm25_m_esd,recomp_ext_m_esd,trunc_m_esd,ecd,ecd_unit,ecdi0,ecdrand,cnech,ftceb,ftceb_unit"
BASE="full,closedbook,provence,llmlingua2,longllmlingua,llm_summary,ftce_r4,bge_r4,bm25_r4,recomp_ext_r4,trunc_r4"
run x_r7      $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --methods "{$NEW,$BASE}"
run x_r7p     $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --datasets $P --methods "{esd,ecd,ecd_unit,ftceb,ftceb_unit}"
run x_ev7     $PY src/evaluate.py --reader qwen7b
run x_rl      $PY src/reader.py --name llama8b --model models/Llama-3.1-8B-Instruct --quant bitsandbytes --methods "{$NEW,$BASE}"
run x_evl     $PY src/evaluate.py --reader llama8b
run x_r3      $PY src/reader.py --name qwen3b --model models/Qwen2.5-3B-Instruct-AWQ --methods "{$NEW,$BASE}"
run x_ev3     $PY src/evaluate.py --reader qwen3b
touch stages/esd_all.done; echo ESD_ALL_DONE
