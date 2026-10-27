#!/bin/bash
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=6
P=musique_k5,musique_k10,nq_k5,nq_k10; O=hotpotqa_open,musique_open
run p2_pecd   $PY src/ecd_run.py --ckpt ckpt/ecd --ctx chain --name ecd --datasets $P,$O --ratios "" --taus ""
run p2_pftce  $PY src/ecd_run.py --ckpt ckpt/ftce --ctx none --name ftceb --datasets $P --ratios "" --taus ""
touch stages/par2_all.done; echo PAR2_DONE
