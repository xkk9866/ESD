"""Read-only: aggregate results/preds/<reader>/<ds>/*.jsonl into JSON on stdout (does not touch results/summary_*.json)."""
import glob, json, os, sys
sys.path.insert(0, "src")
import numpy as np
from metrics import fidelity

out = {}
for rd in sorted(os.listdir("results/preds")):
    for dsdir in sorted(glob.glob(f"results/preds/{rd}/*")):
        ds = os.path.basename(dsdir)
        tp = f"data/proc/test_{ds}.jsonl"
        test = {json.loads(l)["id"]: json.loads(l) for l in open(tp, encoding="utf-8")} if os.path.exists(tp) else {}
        for fp in sorted(glob.glob(f"{dsdir}/*.jsonl")):
            m = os.path.basename(fp)[:-6]
            try:
                P = [json.loads(l) for l in open(fp)]
            except Exception:
                continue
            if not P:
                continue
            row = {"EM": 100 * np.mean([p["em"] for p in P]), "F1": 100 * np.mean([p["f1"] for p in P]),
                   "Acc": 100 * np.mean([p.get("acc", 0) for p in P]), "Tok": float(np.mean([p["ctx_tokens"] for p in P])),
                   "N": len(P), "CompMs": 1000 * float(np.mean([p.get("comp_time", 0) or 0 for p in P]))}
            cp = f"results/compressed/{ds}/{m}.jsonl"
            if test and os.path.exists(cp) and "--fid" in sys.argv:
                C = {json.loads(l)["id"]: json.loads(l) for l in open(cp, encoding="utf-8")}
                try:
                    fid = [fidelity(test[p["id"]], C[p["id"]]["context"], C[p["id"]].get("kept")) for p in P if p["id"] in C]
                    qps = [f["qp"] for f in fid if f["qp"] is not None]
                    row.update(SF=100 * np.mean([f["sf_rec"] for f in fid]), AnsRec=100 * np.mean([f["ans_rec"] for f in fid]),
                               QP=100 * np.mean(qps) if qps else None, Prov=100 * np.mean([f["prov"] for f in fid]))
                except Exception as e:
                    row["fid_err"] = str(e)[:80]
            out.setdefault(rd, {}).setdefault(ds, {})[m] = row
json.dump(out, sys.stdout)
