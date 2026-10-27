"""Mechanism analysis for ESD (results/esd_analysis.json):
  sets  : for every extractive method, units kept, evidence precision / recall, exact-cover rate (S* within E),
          Spearman rho between |E| and |S*|, and per hop count on MuSiQue
  halt  : for ESD chains, the END probability at every visited state vs. whether the state is already sufficient
          (AUROC, reliability bins), and the rate at which the step cap was hit
  pool  : the same set statistics on pools of 5 / 10 / 20 passages (pool-size robustness of the halting rule)
  sig   : paired bootstrap of EM between ESD and every other method (per reader, per dataset)"""
import argparse, json, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from esd import units, gold_of


def load_ex(ds):
    return {json.loads(l)["id"]: json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")}


def set_stats(ds, name, ex):
    fp = f"results/compressed/{ds}/{name}.jsonl"
    if not os.path.exists(fp):
        return None
    ng, nk, pr, rc, cov, hop = [], [], [], [], [], {}
    for l in open(fp, encoding="utf-8"):
        r = json.loads(l); e = ex[r["id"]]
        if r.get("kept") is None:
            return None
        us = units(e); idx = {(pi, si): i for i, (pi, si, _) in enumerate(us)}
        k = {idx[tuple(x)] for x in r["kept"]}; g = set(gold_of(e, us))
        if not g:
            continue
        ng.append(len(g)); nk.append(len(k)); pr.append(len(g & k) / max(1, len(k))); rc.append(len(g & k) / len(g)); cov.append(float(g <= k))
        if ds.startswith("musique"):
            hop.setdefault(r["id"].split("hop")[0], []).append((len(k), len(g), rc[-1], cov[-1]))
    d = {"units": float(np.mean(nk)), "gold": float(np.mean(ng)), "prec": 100 * float(np.mean(pr)), "rec": 100 * float(np.mean(rc)),
         "cover": 100 * float(np.mean(cov)), "rho": float(spearmanr(ng, nk).correlation), "n": len(nk)}
    if hop:
        d["hops"] = {h: {"n": len(v), "units": float(np.mean([x[0] for x in v])), "gold": float(np.mean([x[1] for x in v])),
                         "rec": 100 * float(np.mean([x[2] for x in v])), "cover": 100 * float(np.mean([x[3] for x in v]))} for h, v in sorted(hop.items())}
    return d


def halt_stats(ds, name, ex):
    fp = f"results/chains/{ds}/{name}.jsonl"
    if not os.path.exists(fp):
        return None
    y, p, capped = [], [], []
    for l in open(fp, encoding="utf-8"):
        r = json.loads(l); e = ex[r["id"]]; g = set(gold_of(e))
        if not g:
            continue
        C = set()
        for t, (u, pu, pend) in enumerate(r["chain"]):
            if t > 0:  # END is an action only in non-empty states
                y.append(float(g <= C)); p.append(pend)
            C.add(u)
        if r["p_end"] is not None:
            y.append(float(g <= C)); p.append(r["p_end"])
        capped.append(r["p_end"] is None)
    y, p = np.array(y), np.array(p)
    bins = [[int(((p >= b / 10) & (p < (b + 1) / 10 + (b == 9))).sum()),
             float(y[(p >= b / 10) & (p < (b + 1) / 10 + (b == 9))].mean()) if ((p >= b / 10) & (p < (b + 1) / 10 + (b == 9))).any() else None] for b in range(10)]
    return {"auroc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None, "states": len(y), "suff_rate": float(y.mean()),
            "bins": bins, "ece": float(sum(abs((i + .5) / 10 - b[1]) * b[0] for i, b in enumerate(bins) if b[0]) / len(y)),
            "capped": 100 * float(np.mean(capped))}


def bootstrap(a, b, B=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(a) - np.asarray(b); n = len(d)
    s = d[rng.integers(0, n, (B, n))].mean(1)
    return {"diff": 100 * float(d.mean()), "lo": 100 * float(np.percentile(s, 2.5)), "hi": 100 * float(np.percentile(s, 97.5)),
            "p": float(min(1.0, 2 * min((s <= 0).mean(), (s >= 0).mean())))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/esd_analysis.json")
    ap.add_argument("--readers", default="qwen7b,llama8b,qwen3b")
    a = ap.parse_args()
    M = ["esd", "esd_marg", "esdr0", "esdrand", "ecd", "ecd_unit", "ecdi0", "ecdrand", "cnech", "ftceb", "ftceb_unit",
         "ftce_ok", "ftce_m_esd", "bge_m_esd", "bm25_m_esd", "recomp_ext_m_esd", "ftce_r4", "bge_r4", "adacomp",
         "esdrd", "esdsf6k", "esds1", "esds1_marg", "ftces1_m_esds1"]
    out = {"sets": {}, "halt": {}, "pool": {}, "sig": {}, "hop_em": {}}
    for m in M:  # MuSiQue EM by hop count (main reader)
        fp = f"results/preds/qwen7b/musique/{m}.jsonl"
        if os.path.exists(fp):
            H = {}
            for l in open(fp):
                r = json.loads(l); H.setdefault(r["id"].split("hop")[0], []).append(r["em"])
            out["hop_em"][m] = {h: 100 * float(np.mean(v)) for h, v in sorted(H.items())}
    for ds in ["nq", "hotpotqa", "musique", "2wiki", "hotpotqa_open", "musique_open"]:
        if not os.path.exists(f"data/proc/test_{ds}.jsonl"):
            continue
        ex = load_ex(ds)
        out["sets"][ds] = {m: s for m in M if (s := set_stats(ds, m, ex))}
        out["halt"][ds] = {m: h for m in ["esd", "esdr0", "esdrand", "esds1", "esdrd"] if (h := halt_stats(ds, m, ex))}
        # cost: compressor seconds per question (as logged by each compressor) and reader prefill tokens
        cost = {}
        for m in ["esd", "ftce_r4", "ftce_m_esd", "adacomp", "ecd", "provence", "llmlingua2", "longllmlingua", "llm_summary", "recomp_abs", "exit", "compact"]:
            fp = f"results/compressed/{ds}/{m}.jsonl"
            if os.path.exists(fp):
                T = [json.loads(l).get("time", 0.0) for l in open(fp, encoding="utf-8")]
                cost[m] = {"sec": float(np.mean(T)), "p90": float(np.percentile(T, 90))}
        cs = f"results/chains/{ds}/esd.jsonl"
        if os.path.exists(cs):
            R = [json.loads(l) for l in open(cs)]
            cost["esd_passes"] = float(np.mean([len(r["chain"]) + (r["p_end"] is not None) for r in R]))
        out.setdefault("cost", {})[ds] = cost
        for m, s in out["sets"][ds].items():
            print(ds, m, {k: round(v, 2) for k, v in s.items() if isinstance(v, float)}, flush=True)
        print(ds, "halt", {m: (round(h["auroc"] or 0, 3), round(h["ece"], 3), h["capped"]) for m, h in out["halt"][ds].items()}, flush=True)
        for rd in a.readers.split(","):
            P = {}
            for m in os.listdir(f"results/preds/{rd}/{ds}") if os.path.isdir(f"results/preds/{rd}/{ds}") else []:
                P[m[:-6]] = {json.loads(l)["id"]: json.loads(l)["em"] for l in open(f"results/preds/{rd}/{ds}/{m}")}
            if "esd" not in P:
                continue
            ids = list(P["esd"])
            out["sig"].setdefault(rd, {})[ds] = {m: bootstrap([P["esd"][i] for i in ids], [v[i] for i in ids])
                                                for m, v in P.items() if m != "esd" and all(i in v for i in ids)}
    for base in ["musique", "nq"]:
        for k in [5, 10, 20]:
            ds = base if k == 20 else f"{base}_k{k}"
            if not os.path.exists(f"data/proc/test_{ds}.jsonl"):
                continue
            ex = load_ex(ds)
            out["pool"].setdefault(base, {})[k] = {m: s for m in ["esd", "ecd", "ecd_unit", "ftceb", "ftceb_unit"] if (s := set_stats(ds, m, ex))}
            print("pool", base, k, {m: (round(s["units"], 2), round(s["rec"], 1), round(s["prec"], 1)) for m, s in out["pool"][base][k].items()}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
