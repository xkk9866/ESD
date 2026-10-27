"""Cross-unit fidelity evaluation.

* DRR (dependency retention rate): among annotated dependencies "unit a cannot be read correctly without unit b"
  (LLM-annotated on the test split, types: reference / qualification), the fraction of cases with a retained
  whose context b is also retained. Reported per type.
* Bridge completeness: fraction of questions for which every gold passage keeps at least one gold unit
  (all hops of the reasoning chain survive compression).
* Annotator agreement (Cohen's kappa) between the two LLM annotators on the test pairs."""
import argparse, glob, json, os
import numpy as np
from metrics import words, sent_retained


def load_deps(ds, tag):
    p = f"data/proc/deps_test_{ds}_{tag}.jsonl"
    return {json.loads(l)["id"]: json.loads(l)["pairs"] for l in open(p, encoding="utf-8")} if os.path.exists(p) else {}


def kept_set(ex, rec):
    if rec.get("kept") is not None:
        return set(map(tuple, rec["kept"]))
    cset = set(words(rec["context"]))
    return {(pi, si) for pi, p in enumerate(ex["passages"]) for si, s in enumerate(p["sents"]) if sent_retained(s, cset)}


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in set(a) | set(b))
    return (po - pe) / (1 - pe + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="nq,hotpotqa,musique,2wiki")
    ap.add_argument("--methods", default="*")
    a = ap.parse_args()
    out = {}
    for ds in a.datasets.split(","):
        test = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")}
        dq, dl = load_deps(ds, "qwen14b"), load_deps(ds, "llama8b")
        # agreement on binary dependency (1/2 vs 3) and on 3-way type
        A, B, A3, B3 = [], [], [], []
        for i, pairs in dq.items():
            m = {(x[0], x[1], x[2]): x[3] for x in dl.get(i, [])}
            for pi, s_a, s_b, lab in pairs:
                if (pi, s_a, s_b) in m:
                    A.append(int(lab in (1, 2))); B.append(int(m[(pi, s_a, s_b)] in (1, 2)))
                    A3.append(lab); B3.append(m[(pi, s_a, s_b)])
        res = {"agreement": {"n": len(A), "acc": float(np.mean(np.array(A) == np.array(B))) if A else None,
                             "kappa_bin": float(kappa(A, B)) if A else None, "kappa_3way": float(kappa(A3, B3)) if A else None,
                             "dep_rate": float(np.mean(A)) if A else None},
               "methods": {}}
        pats = a.methods.strip("{}").split(",")
        files = sorted(set(f for pt in pats for f in glob.glob(f"results/compressed/{ds}/{pt}.jsonl")))
        for fp in files:
            meth = os.path.basename(fp)[:-6]
            recs = [json.loads(l) for l in open(fp, encoding="utf-8")]
            ret = {1: [0, 0], 2: [0, 0]}
            bridge = []
            for r in recs:
                ex = test[r["id"]]
                K = kept_set(ex, r)
                for pi, s_a, s_b, lab in dq.get(r["id"], []):
                    if lab in (1, 2) and (pi, s_a) in K:
                        ret[lab][1] += 1; ret[lab][0] += int((pi, s_b) in K)
                gp = [pi for pi, p in enumerate(ex["passages"]) if any(p["gold"])]
                bridge.append(all(any((pi, si) in K for si, g in enumerate(ex["passages"][pi]["gold"]) if g) for pi in gp))
            tot = [ret[1][0] + ret[2][0], ret[1][1] + ret[2][1]]
            res["methods"][meth] = {"DRR": 100 * tot[0] / max(1, tot[1]), "DRR_ref": 100 * ret[1][0] / max(1, ret[1][1]),
                                    "DRR_qual": 100 * ret[2][0] / max(1, ret[2][1]), "n_dep": tot[1],
                                    "bridge": 100 * float(np.mean(bridge))}
        out[ds] = res
        print(ds, json.dumps(res["agreement"]), flush=True)
    json.dump(out, open("results/dep_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
