#!/bin/bash
# parallel side queue on GPU 6 (light jobs); the main queue's copies of these stages are pre-marked done
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=6
D4=nq,hotpotqa,musique,2wiki; P=musique_k5,musique_k10,nq_k5,nq_k10; O=hotpotqa_open,musique_open
run p_exitce  $PY src/exit_ce.py --mode train
run p_exitcei $PY src/exit_ce.py --mode infer --datasets $D4,$O
run p_rabs    $PY src/extra_baselines.py --method recomp_abs --datasets $D4
run p_pesd    $PY src/esd_run.py --ckpt ckpt/esd --name esd --datasets $P
run p_pecd    $PY src/ecd_run.py --ckpt ckpt/ecd --ctx chain --name ecd --datasets $P --ratios "" --taus ""
run p_pftce   $PY src/ecd_run.py --ckpt ckpt/ftce --ctx none --name ftceb --datasets $P --ratios "" --taus ""
touch stages/par_all.done; echo PAR_DONE
