import json, sys, numpy as np
name = sys.argv[1] if len(sys.argv) > 1 else "esd"
for ds in ["nq", "hotpotqa", "musique", "2wiki", "hotpotqa_open", "musique_open"]:
    try:
        R = [json.loads(l) for l in open(f"results/chains/{ds}/{name}.jsonl")]
    except FileNotFoundError:
        continue
    pe = np.array([r["p_end"] if r["p_end"] is not None else np.nan for r in R])
    L = np.array([len(r["chain"]) for r in R])
    # probability mass of the best unit at the stop step is unknown; report p_end at stop
    print(f"{ds:14s} n={len(R)} len {L.mean():.2f} capped {np.isnan(pe).mean()*100:.1f}% "
          f"p_end<0.5 {np.nanmean(pe<0.5)*100:.1f}% p_end<0.8 {np.nanmean(pe<0.8)*100:.1f}% median {np.nanmedian(pe):.3f} "
          f"len-hist {np.bincount(L, minlength=6)[:8].tolist()}")
