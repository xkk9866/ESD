import json, os, sys
import numpy as np
sys.path.insert(0, "src")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def boot(a, b, R=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    m = d[np.random.default_rng(seed).integers(0, len(d), (R, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5), min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))


pairs = [("qwen7b", "nq", "spa_rd", "full"), ("qwen7b", "nq", "spa_rd", "provence"), ("qwen7b", "nq", "spa_rd", "spa_sf"), ("qwen7b", "nq", "spa_rd", "esd"),
         ("qwen7b", "nq", "spa_rd", "exitce"), ("qwen7b", "nq", "spa_rd", "adacomp"),
         ("qwen7b", "hotpotqa", "spa", "esd"), ("qwen7b", "musique", "spa", "esd"), ("qwen7b", "2wiki", "spa", "esd"), ("qwen7b", "musique", "spa", "spa_add"),
         ("qwen7b", "musique", "spa", "spa_norm"), ("qwen7b", "hotpotqa_open", "spa", "esd"), ("llama8b", "musique", "spa", "esd"), ("qwen3b", "musique", "spa", "esd"),
         ("qwen7b", "musique", "spa", "exitce"), ("qwen7b", "hotpotqa", "spa", "exitce"), ("qwen7b", "2wiki", "spa", "exitce"), ("qwen7b", "musique", "spar0", "spa"),
         ("qwen7b", "hotpotqa", "spa_sf", "spa_rd"), ("qwen7b", "musique", "spa_sf", "spa_rd"), ("qwen7b", "2wiki", "spa_sf", "spa_rd")]
for rd, ds, a, b in pairs:
    A = {r["id"]: r["em"] for r in jl(f"results/preds/{rd}/{ds}/{a}.jsonl")}
    B = {r["id"]: r["em"] for r in jl(f"results/preds/{rd}/{ds}/{b}.jsonl")}
    ids = [i for i in A if i in B]
    if ids:
        print(rd, ds, a, "vs", b, "n=%d diff %.1f [%.1f, %.1f] p=%.3f" % ((len(ids),) + boot([A[i] for i in ids], [B[i] for i in ids])))
