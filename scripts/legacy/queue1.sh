#!/bin/bash
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=1
run b_summary    $PY src/reader.py --mode summarize
run train_ftce   $PY src/ftce_train.py --out ckpt/ftce
waitfor models_ready
run b_lingua2    $PY src/baselines.py --method llmlingua2
run b_provence   $PY src/baselines.py --method provence
run b_longlingua $PY src/baselines.py --method longllmlingua --ckpt models/Qwen2.5-7B-Instruct
waitfor vc_main
run b_trunc   $PY src/baselines.py --method trunc
run b_bm25    $PY src/baselines.py --method bm25
run b_bge     $PY src/baselines.py --method bge
run b_ftce    $PY src/baselines.py --method ftce --ckpt ckpt/ftce
run b_recomp  $PY src/baselines.py --method recomp_ext --modes match,r2,r4,r8
run r_qwen7b  $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ
run usr       $PY src/nli_usr.py
run eval_qwen7b $PY src/evaluate.py --reader qwen7b
run r_qwen14b $PY src/reader.py --name qwen14b --model models/Qwen2.5-14B-Instruct-AWQ --methods "$MAIN"
run r_llama8b $PY src/reader.py --name llama8b --model models/Llama-3.1-8B-Instruct --quant bitsandbytes --methods "$MAIN"
waitfor q0_all
run r_abl     $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ
run usr2      $PY src/nli_usr.py
run eval_final $PY src/evaluate.py --reader qwen7b
run eval_14b  $PY src/evaluate.py --reader qwen14b
run eval_llama $PY src/evaluate.py --reader llama8b
echo "=== QUEUE1 FINISHED"
