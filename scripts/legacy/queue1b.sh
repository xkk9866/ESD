#!/bin/bash
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=1
A="--n 6000 --max_len 2560"
run b_lingua2    $PY src/baselines.py --method llmlingua2
run b_longlingua $PY src/baselines.py --method longllmlingua --ckpt models/Qwen2.5-7B-Instruct
run train_nover  $PY src/train.py --out ckpt/abl_noverify --no_verify $A
run vc_nover     $PY src/vericomp.py --ckpt ckpt/abl_noverify --name vcabl-noverifytrain --alphas 0.1 --ratios 4 --repair_iters 0 --tau_s -1
run train_eta0   $PY src/train.py --out ckpt/abl_eta0 --eta 0 $A
run vc_eta0      $PY src/vericomp.py --ckpt ckpt/abl_eta0 --name vcabl-eta0 --alphas 0.1 --ratios 4
waitfor models_ready
run b_provence   $PY src/baselines.py --method provence
touch stages/q1b_pre.done
waitfor q0b_base
run usr          $PY src/nli_usr.py
run r_qwen14b    $PY src/reader.py --name qwen14b --model models/Qwen2.5-14B-Instruct-AWQ --methods "$MAIN"
waitfor q0b_all
run eval_qwen7b  $PY src/evaluate.py --reader qwen7b
run eval_14b     $PY src/evaluate.py --reader qwen14b
run eval_llama   $PY src/evaluate.py --reader llama8b
echo "=== QUEUE1B FINISHED"
