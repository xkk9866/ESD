"""Mechanism analysis of decoded chains: does the decoded length track how much evidence a question needs?
For each method with Bayes stopping: units kept, evidence precision / recall, exact-cover rate, and the rank
correlation between #units kept and #gold units; on MuSiQue also per hop count (2/3/4). Also reports, per step
of the ECD chain, how often the chosen unit is gold (conditional precision) and the calibration of p."""
import argparse, json
import numpy as np
from scipy.stats import spearmanr
from ecd import units, sufficiency_stop


def gold_of(e):
    return {i for i, (pi, si, _) in enumerate(units(e)) if e["passages"][pi]["gold"][si]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="ecd,cnech,ecdi0,ecdrand,ftceb")
    ap.add_argument("--reader", default="qwen7b")
    ap.add_argument("--out", default="results/ecd_analysis.json")
    a = ap.parse_args()
    out = {}
    for ds in ["nq", "hotpotqa", "musique", "2wiki"]:
        ex = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")}
        out[ds] = {}
        for nm in a.names.split(","):
            try:
                R = [json.loads(l) for l in open(f"results/chains/{ds}/{nm}.jsonl", encoding="utf-8")]
                P = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/preds/{a.reader}/{ds}/{nm}.jsonl", encoding="utf-8")}
            except FileNotFoundError:
                continue
            ng, nk, prec, rec, cover, em, hop = [], [], [], [], [], [], {}
            bins, sb = np.zeros((10, 2)), np.zeros((10, 2))
            for r in R:
                e = ex[r["id"]]; g = gold_of(e); k = sufficiency_stop(r["chain"])
                ng.append(len(g)); nk.append(len(k))
                prec.append(len(g & set(k)) / len(k)); rec.append(len(g & set(k)) / max(1, len(g)))
                cover.append(float(g <= set(k))); em.append(P[r["id"]]["em"] if r["id"] in P else np.nan)
                if ds == "musique":
                    h = r["id"].split("hop")[0]
                    hop.setdefault(h, []).append((len(k), rec[-1], em[-1]))
                chosen = set()
                for u, p, sf in r["chain"]:
                    b = min(9, int(p * 10)); bins[b] += [1, float(u in g - chosen)]
                    sb[min(9, int(sf * 10))] += [1, float(g <= chosen)]; chosen.add(u)
            d = {"units": np.mean(nk), "gold": np.mean(ng), "prec": 100 * np.mean(prec), "rec": 100 * np.mean(rec),
                 "cover": 100 * np.mean(cover), "EM": 100 * np.nanmean(em), "rho": spearmanr(ng, nk).correlation,
                 "calib": [[float(c), float(y / c) if c else None] for c, y in bins],
                 "suff_calib": [[float(c), float(y / c) if c else None] for c, y in sb],
                 "suff_ece": float(sum(abs((i + .5) / 10 - y / c) * c for i, (c, y) in enumerate(sb) if c) / sb[:, 0].sum()),
                 "ece": float(sum(abs((i + .5) / 10 - y / c) * c for i, (c, y) in enumerate(bins) if c) / bins[:, 0].sum())}
            if hop:
                d["hops"] = {h: {"n": len(v), "units": float(np.mean([x[0] for x in v])), "rec": 100 * float(np.mean([x[1] for x in v])),
                                 "EM": 100 * float(np.nanmean([x[2] for x in v]))} for h, v in sorted(hop.items())}
            out[ds][nm] = d
            print(ds, nm, " ".join(f"{k}={v:.2f}" for k, v in d.items() if isinstance(v, float)), d.get("hops", ""), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
