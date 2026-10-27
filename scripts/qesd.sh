#!/bin/bash
# ESD: train (round 0 teacher forcing + round 1 DAgger), GPU 6 only
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=6
run esd_train $PY src/esd.py --out ckpt/esd
echo ESD_TRAIN_DONE
