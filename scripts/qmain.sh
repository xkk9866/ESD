#!/bin/bash
# ESD paper: every experiment after the main training run. GPU 6 only; every LLM is int4 and < 10B.
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=6
waitfor esd_train
D4=nq,hotpotqa,musique,2wiki; P=musique_k5,musique_k10,nq_k5,nq_k10; O=hotpotqa_open,musique_open
# ---------------- A. compressors
run x_esd     $PY src/esd_run.py --ckpt ckpt/esd    --name esd     --datasets $D4,$O
run x_esdr0   $PY src/esd_run.py --ckpt ckpt/esd_r0 --name esdr0   --datasets $D4
run x_esdrand $PY src/esd_run.py --ckpt ckpt/esd    --name esdrand --ctx random --datasets $D4
run x_mftce   $PY src/baselines.py --method ftce --ckpt ckpt/ftce --modes m_esd,ok,r4 --datasets $D4,$O
run x_mbge    $PY src/baselines.py --method bge --modes m_esd --datasets $D4,$O
run x_mbm25   $PY src/baselines.py --method bm25 --modes m_esd --datasets $D4
run x_mrec    $PY src/baselines.py --method recomp_ext --modes m_esd --datasets $D4
run x_mtr     $PY src/baselines.py --method trunc --modes m_esd --datasets $D4
run x_ada     $PY src/adacomp.py --mode train
run x_adai    $PY src/adacomp.py --mode infer --datasets $D4,$O
run x_ofull   $PY src/baselines.py --method full --datasets $O
run x_oprov   $PY src/baselines.py --method provence --datasets $O
run x_oll2    $PY src/baselines.py --method llmlingua2 --datasets $O
run x_oecd    $PY src/ecd_run.py --ckpt ckpt/ecd --ctx chain --name ecd --datasets $O --ratios "" --taus ""
run x_pesd    $PY src/esd_run.py --ckpt ckpt/esd --name esd --datasets $P
run x_pecd    $PY src/ecd_run.py --ckpt ckpt/ecd --ctx chain --name ecd --datasets $P --ratios "" --taus ""
run x_pftce   $PY src/ecd_run.py --ckpt ckpt/ftce --ctx none --name ftceb --datasets $P --ratios "" --taus ""
run x_rabs    $PY src/extra_baselines.py --method recomp_abs --datasets $D4
[ -f models/gemma-2b-it/config.json ] && run x_exit $PY src/extra_baselines.py --method exit --datasets $D4
run x_exitce  $PY src/exit_ce.py --mode train
run x_exitcei $PY src/exit_ce.py --mode infer --datasets $D4,$O
run x_compact $PY src/extra_baselines.py --method compact --datasets $D4
# ---------------- B. main reader
NEW="esd,esd_marg,esdr0,esdr0_marg,esdrand,ftce_m_esd,ftce_ok,bge_m_esd,bm25_m_esd,recomp_ext_m_esd,trunc_m_esd,adacomp,exitce,recomp_abs,exit,compact,compact_qwen,ecd,ecd_unit,ecdi0,ecdrand,cnech,ftceb,ftceb_unit"
BASE="full,closedbook,provence,provence_q*,llmlingua2,llmlingua2_q*,longllmlingua,longllmlingua_q*,llm_summary,ftce_r2,ftce_r4,ftce_r8,bge_r4,bm25_r4,recomp_ext_r4,trunc_r4"
run x_r7      $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --datasets $D4 --methods "{$NEW,$BASE}"
run x_r7o     $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --datasets $O --max_model_len 16384 --methods "{full,esd,esd_marg,ftce_m_esd,ftce_r4,bge_m_esd,adacomp,exitce,provence,llmlingua2,ecd}"
run x_r7p     $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --datasets $P --methods "{esd,ecd,ecd_unit,ftceb,ftceb_unit}"
run x_ev7     $PY src/evaluate.py --reader qwen7b --datasets $D4,$O,$P
run x_an      $PY src/esd_analysis.py
# ---------------- C. supervision without supporting-fact labels, and seeds
run z_orac    $PY src/reader_oracle.py
run z_trd     $PY src/esd.py --train data/proc/train_reader.jsonl --out ckpt/esd_rd --hard_cache ckpt/esd_rd_hard.json --dagger_n 100000
run z_tsf     $PY src/esd.py --train data/proc/train_sfsub.jsonl --out ckpt/esd_sf6k --hard_cache ckpt/esd_sf6k_hard.json --dagger_n 100000
run z_drd     $PY src/esd_run.py --ckpt ckpt/esd_rd   --name esdrd  --datasets $D4
run z_dsf     $PY src/esd_run.py --ckpt ckpt/esd_sf6k --name esdsf6k --datasets $D4
run z_s1      $PY src/esd.py --out ckpt/esd_s1 --seed 1
run z_ds1     $PY src/esd_run.py --ckpt ckpt/esd_s1 --name esds1 --datasets $D4
run z_ft1     $PY src/ftce_train.py --out ckpt/ftce_s1 --seed 1
run z_mft1    $PY src/baselines.py --method ftce --ckpt ckpt/ftce_s1 --name ftces1 --modes m_esds1,r4 --datasets $D4
run z_r7      $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --datasets $D4 --methods "{esdrd,esdrd_marg,esdsf6k,esds1,esds1_marg,ftces1_m_esds1,ftces1_r4}"
run z_ev7     $PY src/evaluate.py --reader qwen7b --datasets $D4,$O,$P
# ---------------- D. reader generalization
RG="{$NEW,$BASE,esdrd,esdsf6k,esds1}"
run w_rl      $PY src/reader.py --name llama8b --model models/Llama-3.1-8B-Instruct --quant bitsandbytes --datasets $D4 --methods "$RG"
run w_r3      $PY src/reader.py --name qwen3b --model models/Qwen2.5-3B-Instruct-AWQ --datasets $D4 --methods "$RG"
run w_evl     $PY src/evaluate.py --reader llama8b
run w_ev3     $PY src/evaluate.py --reader qwen3b
run w_usr     $PY src/nli_usr.py --methods esd,adacomp,exitce,ftce_m_esd,ftce_r4,trunc_r4,provence,llmlingua2,longllmlingua,llm_summary,recomp_abs,exit,compact,compact_qwen
run w_ev7     $PY src/evaluate.py --reader qwen7b --datasets $D4,$O,$P
run w_an      $PY src/esd_analysis.py
touch stages/main_all.done; echo MAIN_ALL_DONE
