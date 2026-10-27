"""Aggregate answer quality, fidelity, provenance and cost into results/summary_<reader>.json"""
import argparse, glob, json, os
import numpy as np
from metrics import fidelity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="qwen7b")
    ap.add_argument("--datasets", default="nq,hotpotqa,musique,2wiki")
    a = ap.parse_args()
    usr = json.load(open("results/usr.json")) if os.path.exists("results/usr.json") else {}
    summ = {}
    for ds in a.datasets.split(","):
        test = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")}
        full_tok = None
        fp_full = f"results/preds/{a.reader}/{ds}/full.jsonl"
        if os.path.exists(fp_full):
            full_tok = np.mean([json.loads(l)["ctx_tokens"] for l in open(fp_full)])
        for fp in sorted(glob.glob(f"results/preds/{a.reader}/{ds}/*.jsonl")):
            meth = os.path.basename(fp)[:-6]
            P = [json.loads(l) for l in open(fp)]
            C = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/compressed/{ds}/{meth}.jsonl", encoding="utf-8")}
            fid = [fidelity(test[p["id"]], C[p["id"]]["context"], C[p["id"]].get("kept")) for p in P]
            qps = [f["qp"] for f in fid if f["qp"] is not None]
            tok = np.mean([p["ctx_tokens"] for p in P])
            row = {"EM": 100 * np.mean([p["em"] for p in P]), "F1": 100 * np.mean([p["f1"] for p in P]),
                   "Acc": 100 * np.mean([p["acc"] for p in P]), "Tok": tok,
                   "Ratio": (full_tok / tok) if (full_tok and tok > 0) else None,
                   "SF": 100 * np.mean([f["sf_rec"] for f in fid]), "AnsRec": 100 * np.mean([f["ans_rec"] for f in fid]),
                   "QP": 100 * np.mean(qps) if qps else None, "Prov": 100 * np.mean([f["prov"] for f in fid]),
                   "CompMs": 1000 * np.mean([p["comp_time"] for p in P]), "N": len(P),
                   "Fallback": 100 * np.mean([C[p["id"]].get("fallback", False) for p in P]),
                   "USR": usr.get(ds, {}).get(meth)}
            summ.setdefault(ds, {})[meth] = row
    json.dump(summ, open(f"results/summary_{a.reader}.json", "w"), indent=1)
    for ds, rows in summ.items():
        print(f"\n## {ds}")
        print("method | EM | F1 | Tok | x | SF | AnsRec | QP | Prov | ms | FB")
        for m, r in sorted(rows.items(), key=lambda x: x[0]):
            f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"
            print(f"{m} | {f(r['EM'])} | {f(r['F1'])} | {f(r['Tok'],0)} | {f(r['Ratio'])} | {f(r['SF'])} | {f(r['AnsRec'])} | "
                  f"{f(r['QP'])} | {f(r['Prov'])} | {f(r['CompMs'],0)} | {f(r['Fallback'])}")


if __name__ == "__main__":
    main()
