import json, sys, numpy as np
sys.path.insert(0, "src")
from esd import units, gold_of
name = sys.argv[1] if len(sys.argv) > 1 else "esd"
for ds in ["nq", "hotpotqa", "musique", "2wiki", "hotpotqa_open", "musique_open"]:
    try:
        R = [json.loads(l) for l in open(f"results/chains/{ds}/{name}.jsonl")]
    except FileNotFoundError:
        continue
    ex = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")}
    pu = [c[1] for r in R for c in r["chain"]]
    cov, ng, prec, wrongp, rightp = [], [], [], [], []
    for r in R:
        e = ex[r["id"]]; g = set(gold_of(e))
        k = [c[0] for c in r["chain"]]
        if not g:
            continue
        cov.append(g <= set(k)); ng.append(len(g)); prec.append(len(g & set(k)) / len(k))
        for c in r["chain"]:
            (rightp if c[0] in g else wrongp).append(c[1])
    pu = np.array(pu)
    print(f"{ds:14s} p(unit)<0.5 {np.mean(pu<0.5)*100:.1f}% <0.9 {np.mean(pu<0.9)*100:.1f}% | cover {np.mean(cov)*100:.1f} prec {np.mean(prec)*100:.1f} "
          f"|S*| {np.mean(ng):.2f} | p(unit) right {np.mean(rightp):.3f} wrong {np.mean(wrongp):.3f} n_wrong {len(wrongp)}")
