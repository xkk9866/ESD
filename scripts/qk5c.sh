#!/bin/bash
# Lane C (GPU 5): reader-defined supervision (QA pairs only) vs SF supervision on the same questions
source ~/research/qrag/queue_common.sh; export CUDA_VISIBLE_DEVICES=5 VLLM_UTIL=0.40
D4=nq,hotpotqa,musique,2wiki
run z_orac  $PY src/reader_oracle.py
waitfor k5a_all; waitfor k5b_all
run z_trd   $PY src/esd.py --train data/proc/train_reader.jsonl --out ckpt/esd_rd   --hard_cache ckpt/esd_rd_hard.json   --rounds 1
run z_tsf   $PY src/esd.py --train data/proc/train_sfsub.jsonl  --out ckpt/esd_sf6k --hard_cache ckpt/esd_sf6k_hard.json --rounds 1
run z_drd   $PY src/psd.py --ckpt ckpt/esd_rd_r0   --name psdrd  --datasets $D4 --levels ""
run z_dsf   $PY src/psd.py --ckpt ckpt/esd_sf6k_r0 --name psdsf  --datasets $D4 --levels ""
run z_r7    $PY src/reader.py --name qwen7b --model models/Qwen2.5-7B-Instruct-AWQ --datasets $D4 --methods "{psdrd,psdrd_gx,psdrd_marg,psdsf,psdsf_gx,psdsf_marg}"
touch stages/k5c_all.done; echo K5C_DONE
