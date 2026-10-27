#!/bin/bash
# SPA queues, GPU 7 only. A: main model; B: no-remove ablation then reader-defined supervision.
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=7
R7='--name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ'
if [ "$1" = A ]; then
run s_spa   $PY src/spa.py --out ckpt/spa --rounds 2 --dagger_n 8000
run s_dspa  $PY src/spa_run.py --ckpt ckpt/spa --name spa
run s_dspa0 $PY src/spa_run.py --ckpt ckpt/spa_r0 --name spar0
run s_dspaT $PY src/spa_run.py --ckpt ckpt/spa --name spa_norm --remove 0
for d in hotpotqa_open musique_open musique_k5 musique_k10 nq_k5 nq_k10; do run s_o_$d $PY src/spa_run.py --ckpt ckpt/spa --name spa --datasets $d; done
touch stages/spaA_dec.done
fi
if [ "$1" = B ]; then
run s_span  $PY src/spa.py --out ckpt/spa_add --rounds 2 --dagger_n 8000 --remove 0
run s_dspan $PY src/spa_run.py --ckpt ckpt/spa_add --name spa_add --remove 0
# (no wait: run alongside decoding)
run s_ro    env VLLM_UTIL=0.55 $PY src/reader_oracle.py --per_ds 2000
run s_spar  $PY src/spa.py --train data/proc/train_reader.jsonl --out ckpt/spa_rd --rounds 2 --dagger_n 4000 --hard_cache ckpt/rd_hard.json
run s_spas  $PY src/spa.py --train data/proc/train_sfsub.jsonl --out ckpt/spa_sf --rounds 2 --dagger_n 3000 --hard_cache ckpt/sf_hard.json
run s_dspar $PY src/spa_run.py --ckpt ckpt/spa_rd --name spa_rd
run s_dspas $PY src/spa_run.py --ckpt ckpt/spa_sf --name spa_sf
touch stages/spaB_dec.done
fi
if [ "$1" = C ]; then
# (no wait: run alongside decoding)
run s_r7a env VLLM_UTIL=0.55 $PY src/reader.py $R7 --methods "{spa,spa_marg,spar0,spa_norm}"
run s_r7o env VLLM_UTIL=0.55 $PY src/reader.py $R7 --datasets hotpotqa_open --methods "{spa}"
waitfor spaB_dec
run s_r7b env VLLM_UTIL=0.55 $PY src/reader.py $R7 --methods "{spa_add,spa_rd,spa_sf}" --datasets nq,hotpotqa,musique,2wiki
run s_rL  env VLLM_UTIL=0.55 $PY src/reader.py --name llama8b --model models/Llama-3.1-8B-Instruct --quant bitsandbytes --methods "{spa}"
run s_r3  env VLLM_UTIL=0.55 $PY src/reader.py --name qwen3b --model models/Qwen2.5-3B-Instruct-AWQ --methods "{spa}"
for r in qwen7b llama8b qwen3b; do run s_ev_$r $PY src/evaluate.py --reader $r; done
touch stages/spa_all.done
fi
