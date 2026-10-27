#!/bin/bash
# second seed of ESD and of FT-CE (GPU 6), run beside the main queue
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=6
D4=nq,hotpotqa,musique,2wiki
run s1_esd   $PY src/esd.py --out ckpt/esd_s1 --seed 1
run s1_desd  $PY src/esd_run.py --ckpt ckpt/esd_s1 --name esds1 --datasets $D4
run s1_ftce  $PY src/ftce_train.py --out ckpt/ftce_s1 --seed 1
run s1_mft   $PY src/baselines.py --method ftce --ckpt ckpt/ftce_s1 --name ftces1 --modes m_esds1,r4 --datasets $D4
touch stages/seed_all.done; echo SEED_DONE
