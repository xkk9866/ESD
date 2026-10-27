"""Read-only: pool statistics for Table 1 (passages, sentence units, words, gold units, gold-passage recall)."""
import json, sys
sys.path.insert(0, "src")
import numpy as np
from esd import units, gold_of

for ds in ["nq", "hotpotqa", "musique", "2wiki", "hotpotqa_open", "musique_open", "musique_k5", "musique_k10", "nq_k5", "nq_k10"]:
    try:
        E = [json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")]
    except FileNotFoundError:
        continue
    P = [len(e["passages"]) for e in E]; U = [len(units(e)) for e in E]
    Wd = [sum(len(s.split()) for p in e["passages"] for s in p["sents"]) for e in E]
    G = [len(gold_of(e)) for e in E]
    print(ds, len(E), "psg %.1f units %.1f words %.0f gold %.2f nogold %.1f%%" % (np.mean(P), np.mean(U), np.mean(Wd), np.mean([g for g in G if g]), 100 * np.mean([g == 0 for g in G])))
n = sum(1 for _ in open("data/proc/train.jsonl", encoding="utf-8"))
print("train", n)
