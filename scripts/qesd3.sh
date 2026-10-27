#!/bin/bash
# after qesd2: fidelity (USR), analysis, tables; then a second training seed of ESD. GPU 6 only.
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=6
waitfor esd_all
run y_usr    $PY src/nli_usr.py --methods esd,esdr0,esd_marg,ftce_m_esd,ecd,ftce_r4,trunc_r4,provence,llmlingua2,longllmlingua,llm_summary
run y_an     $PY src/esd_analysis.py
run y_tab    $PY src/make_tables_esd.py
run y_s1     $PY src/esd.py --out ckpt/esd_s1 --seed 1
run y_ds1    $PY src/esd_run.py --ckpt ckpt/esd_s1 --name esds1
run y_ds1r0  $PY src/esd_run.py --ckpt ckpt/esd_s1_r0 --name esds1r0
run y_rs1    $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --methods "{esds1,esds1_marg,esds1r0}"
run y_ev7    $PY src/evaluate.py --reader qwen7b
touch stages/esd3_all.done; echo ESD3_ALL_DONE
