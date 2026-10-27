"""Certificate and verifier analysis on the test split (labels used only for evaluation):
  * realized omission risk E[L] vs target alpha for the certified outputs <name>_a<alpha>
  * AUROC of the self-verification sufficiency score for detecting lost answer-critical evidence"""
import argparse, json, os
import numpy as np


def passage_miss(ex, kept):
    ks = set(map(tuple, kept))
    gp = {}
    for pi, p in enumerate(ex["passages"]):
        for si, g in enumerate(p["gold"]):
            if g:
                gp.setdefault(pi, []).append((pi, si) in ks)
    return 0.0 if not gp else sum(1 for v in gp.values() if not any(v)) / len(gp)


def auroc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="vc3")
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3")
    ap.add_argument("--datasets", default="nq,hotpotqa,musique,2wiki")
    a = ap.parse_args()
    out = {}
    for ds in a.datasets.split(","):
        test = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")}
        full_w = np.mean([sum(len(s.split()) for p in e["passages"] for s in [p["title"]] + p["sents"]) for e in test.values()])
        row = {}
        for al in a.alphas.split(","):
            fp = f"results/compressed/{ds}/{a.name}_a{al}.jsonl"
            if not os.path.exists(fp):
                continue
            R = [json.loads(l) for l in open(fp, encoding="utf-8")]
            L = [passage_miss(test[r["id"]], r["kept"]) for r in R]
            w = np.mean([len(r["context"].split()) for r in R])
            fb = np.mean([bool(r.get("fallback")) for r in R])
            row[al] = {"risk": float(np.mean(L)), "x": float(full_w / w), "fallback": float(fb)}
        # verifier: sufficiency score of the price-mode selection vs. whether evidence was actually lost
        ver = {}
        for tag in ["p2", "p4", "p8"]:
            fp = f"results/compressed/{ds}/{a.name}_{tag}.jsonl"
            if not os.path.exists(fp):
                continue
            R = [json.loads(l) for l in open(fp, encoding="utf-8")]
            lost = [1 if passage_miss(test[r["id"]], r["kept"]) > 0 else 0 for r in R]
            ver[tag] = {"auroc_insufficient": auroc([-r["suff"] for r in R], lost), "lost_rate": float(np.mean(lost))}
        out[ds] = {"crc": row, "verifier": ver}
        print(ds, json.dumps(out[ds]), flush=True)
    json.dump(out, open(f"results/crc_{a.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
