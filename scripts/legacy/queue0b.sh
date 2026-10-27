#!/bin/bash
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=0
rm -f stages/train_caet.done stages/vc_main.done
run train_caet $PY src/train.py --out ckpt/caet --n 16000 --max_len 2560
run vc_main    $PY src/vericomp.py --ckpt ckpt/caet --name vericomp
run b_trunc    $PY src/baselines.py --method trunc
run b_bm25     $PY src/baselines.py --method bm25
run b_bge      $PY src/baselines.py --method bge
run b_ftce     $PY src/baselines.py --method ftce --ckpt ckpt/ftce
run b_recomp   $PY src/baselines.py --method recomp_ext --modes match,r2,r4,r8
touch stages/q0b_base.done
waitfor q1b_pre
run r_qwen7b   $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ
run r_llama8b  $PY src/reader.py --name llama8b --model models/Llama-3.1-8B-Instruct --quant bitsandbytes --methods "$MAIN"
touch stages/q0b_all.done; echo "=== QUEUE0B FINISHED"
