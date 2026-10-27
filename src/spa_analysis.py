"""Everything the SPA paper reports, from preds / compressed / chains files -> results/spa_analysis.json
  ans[reader][ds][method] = EM, F1, Acc, Tok, N
  ev[ds][method]          = gold-unit recall, exact coverage (S* in E), units kept, precision
  sig[reader][ds][method] = paired bootstrap of EM against SPA (10k resamples)
  climb[ds][name]         = removals, halting, potential of the returned set vs reader correctness (AUROC)
  pool[ds][name]          = units kept on pools of growing size"""
import glob, json, os
import numpy as np
from spa import units, gold_of

DS = ["nq", "hotpotqa", "musique", "2wiki", "hotpotqa_open", "musique_open"]
READERS = ["qwen7b", "llama8b", "qwen3b"]
REF = "spa"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    if y.min() == y.max():
        return None
    r = np.argsort(np.argsort(s)) + 1
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(a, b, R=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(d), (R, len(d)))
    m = d[idx].mean(1)
    return {"diff": 100 * d.mean(), "lo": 100 * np.percentile(m, 2.5), "hi": 100 * np.percentile(m, 97.5),
            "p": float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))}


def main():
    out = {"ans": {}, "ev": {}, "sig": {}, "climb": {}, "pool": {}}
    tests = {ds: {e["id"]: e for e in jl(f"data/proc/test_{ds}.jsonl")} for ds in DS + ["musique_k5", "musique_k10", "nq_k5", "nq_k10"]}
    for rd in READERS:
        for ds in DS:
            P = {os.path.basename(f)[:-6]: {r["id"]: r for r in jl(f)} for f in glob.glob(f"results/preds/{rd}/{ds}/*.jsonl")}
            for m, rows in P.items():
                v = list(rows.values())
                out["ans"].setdefault(rd, {}).setdefault(ds, {})[m] = {
                    "EM": 100 * np.mean([r["em"] for r in v]), "F1": 100 * np.mean([r["f1"] for r in v]),
                    "Acc": 100 * np.mean([r["acc"] for r in v]), "Tok": float(np.mean([r["ctx_tokens"] for r in v])), "N": len(v)}
            if REF in P:
                ids = list(P[REF])
                for m, rows in P.items():
                    if m != REF and all(i in rows for i in ids):
                        out["sig"].setdefault(rd, {}).setdefault(ds, {})[m] = boot([P[REF][i]["em"] for i in ids], [rows[i]["em"] for i in ids])
    for ds in DS:
        T = tests[ds]
        for f in glob.glob(f"results/compressed/{ds}/*.jsonl"):
            m = os.path.basename(f)[:-6]; R = jl(f)
            if not R or "kept" not in R[0]:
                continue
            rec, cov, ln, pr = [], [], [], []
            for r in R:
                e = T.get(r["id"])
                if e is None:
                    continue
                g = {(pi, si) for pi, p in enumerate(e["passages"]) for si, x in enumerate(p["gold"]) if x}
                k = {tuple(x) for x in r["kept"]}
                if not g:
                    continue
                rec.append(len(g & k) / len(g)); cov.append(g <= k); ln.append(len(k)); pr.append(len(g & k) / max(1, len(k)))
            if rec:
                out["ev"].setdefault(ds, {})[m] = {"Rec": 100 * np.mean(rec), "Cov": 100 * np.mean(cov), "Len": float(np.mean(ln)),
                                                   "Prec": 100 * np.mean(pr)}
        for f in glob.glob(f"results/chains/{ds}/spa*.jsonl"):
            name = os.path.basename(f)[:-6]; R = jl(f)
            if not R or "path" not in R[0]:
                continue
            rm = [sum(p[0] == "remove" for p in r["path"]) for r in R]
            row = {"removes_per_q": float(np.mean(rm)), "q_with_remove": 100 * np.mean([x > 0 for x in rm]),
                   "unhalted": 100 * np.mean([not r["halted"] for r in R]), "ms": 1000 * np.mean([r["time"] for r in R]),
                   "moves": float(np.mean([len(r["path"]) + 1 for r in R]))}
            # did removals fix errors? share of removed units that are not gold
            T2 = tests[ds]; good = tot = 0
            for r in R:
                e = T2[r["id"]]; us = units(e); g = set(gold_of(e, us))
                for kind, u, _ in r["path"]:
                    if kind == "remove":
                        tot += 1; good += u not in g
            row["removed_nongold"] = 100 * good / tot if tot else None
            P = {x["id"]: x for x in jl(f"results/preds/qwen7b/{ds}/{name}.jsonl")}
            if P:
                ids = [r["id"] for r in R if r["id"] in P and r["v"] is not None]
                vv = {r["id"]: r["v"] for r in R}
                row["auroc_v_correct"] = auroc([vv[i] for i in ids], [P[i]["em"] or P[i]["acc"] for i in ids])
                # selective answering: EM on the half of questions with the highest potential
                o = sorted(ids, key=lambda i: -vv[i]); h = o[: len(o) // 2]
                row["em_top50"] = 100 * np.mean([P[i]["em"] for i in h]); row["em_all"] = 100 * np.mean([P[i]["em"] for i in ids])
            if ds in ("hotpotqa", "musique", "2wiki", "musique_open", "hotpotqa_open"):
                rs = []; T2 = tests[ds]
                for r in R:
                    e = T2[r["id"]]; g = set(gold_of(e))
                    if g:
                        rs.append((len(r["set"]), len(g)))
                if rs:
                    from scipy.stats import spearmanr
                    row["spearman_len"] = float(spearmanr([a for a, _ in rs], [b for _, b in rs]).correlation)
            out["climb"].setdefault(ds, {})[name] = row
    for base in ["musique", "nq"]:
        for k in ["k5", "k10", ""]:
            ds = f"{base}_{k}" if k else base
            for name in ["spa", "esd"]:
                R = jl(f"results/chains/{ds}/{name}.jsonl")
                if R:
                    L = [len(r["set"]) if "set" in r else len(r["chain"]) for r in R]
                    out["pool"].setdefault(base, {}).setdefault(name, {})[k or "k20"] = float(np.mean(L))
    json.dump(out, open("results/spa_analysis.json", "w"), indent=1)
    for ds in ["nq", "hotpotqa", "musique", "2wiki"]:
        a = out["ans"].get("qwen7b", {}).get(ds, {})
        print(ds, {m: round(a[m]["EM"], 1) for m in sorted(a) if m.startswith(("spa", "esd", "ftce_m_esd", "exitce", "adacomp", "full"))})
    print(json.dumps(out["climb"], indent=0)[:3000])


if __name__ == "__main__":
    main()
