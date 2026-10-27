#!/bin/bash
# extra ablations on GPU 1 once queue1 has finished its main work
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=1
T="--max_len 2560"; A="--n 6000 $T"
waitfor eval_qwen7b
run b_summary   $PY src/reader.py --mode summarize
run train_nover $PY src/train.py --out ckpt/abl_noverify --no_verify $A
run vc_nover    $PY src/vericomp.py --ckpt ckpt/abl_noverify --name vcabl-noverifytrain --alphas 0.1 --ratios 4 --repair_iters 0 --tau_s -1
run train_eta0  $PY src/train.py --out ckpt/abl_eta0 --eta 0 $A
run vc_eta0     $PY src/vericomp.py --ckpt ckpt/abl_eta0 --name vcabl-eta0 --alphas 0.1 --ratios 4
touch stages/q2_all.done; echo "=== QUEUE2 FINISHED"
