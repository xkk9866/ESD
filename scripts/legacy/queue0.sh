#!/bin/bash
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=0
T="--max_len 2560"
run train_caet  $PY src/train.py --out ckpt/caet --n 16000 $T
run vc_main     $PY src/vericomp.py --ckpt ckpt/caet --name vericomp
A="--n 6000 $T"
run train_abfull $PY src/train.py --out ckpt/abl_full $A
run vc_abfull   $PY src/vericomp.py --ckpt ckpt/abl_full --name vcabl-full --alphas 0.1 --ratios 4
run train_norel $PY src/train.py --out ckpt/abl_norel --no_relbias $A
run vc_norel    $PY src/vericomp.py --ckpt ckpt/abl_norel --name vcabl-norel --alphas 0.1 --ratios 4
run train_local $PY src/train.py --out ckpt/abl_local --n_global 0 --no_verify $A
run vc_local    $PY src/vericomp.py --ckpt ckpt/abl_local --n_global 0 --name vcabl-local --alphas 0.1 --ratios 4 --repair_iters 0 --tau_s -1
touch stages/q0_all.done; echo "=== QUEUE0 FINISHED"
