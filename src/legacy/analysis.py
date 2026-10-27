"""Statistical and optimisation analyses requested in review.

1. calibration: expected calibration error (ECE) of raw cost-sensitive scores vs. Bayes-inverted posteriors,
   pooled over units of all test queries (cross-query comparability), from results/dumps/<name>_<ds>.jsonl
2. duality gap: hard-budget optimum (exact MILP, scipy.optimize.milp) vs. the Lagrangian min-cut solution and
   the Lagrangian + residual-fill solution, on the same unary + dependency-penalty objective (couplings off)
3. paired bootstrap confidence intervals for EM differences between methods (same questions, same reader)"""
import argparse, json, os
import numpy as np


# ---------------------------------------------------------------- calibration
def ece(prob, y, bins=15):
    prob, y = np.asarray(prob), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if m.any():
            e += m.mean() * abs(prob[m].mean() - y[m].mean())
    return e


def calibration(name, dss):
    out = {}
    for ds in dss:
        p = f"results/dumps/{name}_{ds}.jsonl"
        if not os.path.exists(p):
            continue
        raw, q, y = [], [], []
        for l in open(p):
            r = json.loads(l)
            for pid, sid, pr, pq, g, ln in r["u"]:
                raw.append(pr); q.append(pq); y.append(g)
        raw, q, y = map(np.asarray, (raw, q, y))
        out[ds] = {"ece_raw": float(ece(raw, y)), "ece_bayes": float(ece(q, y)), "mean_raw": float(raw.mean()),
                   "mean_bayes": float(q.mean()), "base_rate": float(y.mean()), "n_units": int(len(y))}
    return out


# ---------------------------------------------------------------- duality gap
def build(r, dep_thresh=0.0):
    """reconstruct the selection problem of one query from a dump record"""
    U = [(pid, sid) for pid, sid, *_ in r["u"]]
    q = np.array([x[3] for x in r["u"]]); ln = np.array([x[5] for x in r["u"]], dtype=float)
    pids = sorted(set(pid for pid, _ in U))
    return U, q, ln, pids


def milp_opt(q, ln, title_len, pid_of, pairs, B, lam_free=True):
    """exact hard-budget optimum: max sum q m - sum P z, s.t. z_uv >= m_u - m_v, m_u <= t_pid, sum len <= B"""
    from scipy.optimize import milp, LinearConstraint, Bounds
    n, T, E = len(q), len(title_len), len(pairs)
    nv = n + T + E
    c = np.zeros(nv); c[:n] = -q
    for k, (u, v, pen) in enumerate(pairs):
        c[n + T + k] = pen
    A, lo, hi = [], [], []
    for u in range(n):                                  # anchor: m_u - t_p <= 0
        row = np.zeros(nv); row[u] = 1; row[n + pid_of[u]] = -1; A.append(row); lo.append(-np.inf); hi.append(0)
    for k, (u, v, pen) in enumerate(pairs):             # m_u - m_v - z <= 0
        row = np.zeros(nv); row[u] = 1; row[v] = -1; row[n + T + k] = -1; A.append(row); lo.append(-np.inf); hi.append(0)
    row = np.zeros(nv); row[:n] = ln; row[n:n + T] = title_len; A.append(row); lo.append(-np.inf); hi.append(B)
    integ = np.r_[np.ones(n + T), np.zeros(E)]
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=integ, bounds=Bounds(0, 1),
               options={"time_limit": 20})
    return -res.fun if res.status in (0, 1) and res.x is not None else None


def value(m, q, pairs):
    v = float(np.dot(q, m))
    for u, v_, pen in pairs:
        v -= pen * m[u] * (1 - m[v_])
    return v


def duality_gap(name, dss, ratio=4, n_q=150):
    import sys
    sys.path.insert(0, "src")
    from solver import crf_solve_budget, crf_map, cost
    out = {}
    for ds in dss:
        p = f"results/dumps/{name}_{ds}.jsonl"
        if not os.path.exists(p):
            continue
        test = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")}
        gaps_lag, gaps_fill, exact_hits = [], [], 0
        for k, l in enumerate(open(p)):
            if k >= n_q:
                break
            r = json.loads(l); ex = test[r["id"]]
            U, q, ln, pids = build(r)
            pidx = {pid: j for j, pid in enumerate(pids)}
            title_len = np.array([len((ex["passages"][pid]["title"] + ":").split()) for pid in pids], dtype=float)
            pid_of = [pidx[pid] for pid, _ in U]
            uidx = {(pid, sid): i for i, (pid, sid) in enumerate(U)}
            pairs = [(uidx[(pid, a)], uidx[(pid, b)], q[uidx[(pid, a)]] * dv) for pid, a, b, dv in r.get("D", [])
                     if (pid, a) in uidx and (pid, b) in uidx]
            B = (ln.sum() + title_len.sum()) / ratio
            # unit layout for the min-cut solver: [query] + titles + sentences
            n, T = len(U), len(pids)
            pp = [0.0] + [0.0] * T + list(q); lens = [0] + list(title_len) + list(ln)
            deps = [(1 + T + i, 1 + pid_of[i]) for i in range(n)]
            N = 1 + T + n; P = np.zeros((N, N)); C = np.zeros((N, N))
            for u, v, pen in pairs:
                P[1 + T + u, 1 + T + v] += pen
            opt = milp_opt(q, ln, title_len, pid_of, pairs, B)
            if opt is None or opt <= 1e-9:
                continue
            # pure Lagrangian (bisection, no fill)
            lo_, hi_ = 0.0, 50.0; best = None
            for _ in range(25):
                mid = (lo_ + hi_) / 2
                w = [0.0] + [pp[i] - mid * lens[i] / 25.0 for i in range(1, N)]
                kk = crf_map(w, C, deps, P=P)
                if cost(kk, lens) <= B:
                    best, hi_ = kk, mid
                else:
                    lo_ = mid
            mf = crf_solve_budget(pp, C, lens, deps, B, P=P)
            def val(kept):
                m = np.array([1.0 if kept[1 + T + i] else 0.0 for i in range(n)])
                return value(m, q, pairs)
            vl = val(best) if best is not None else 0.0
            vf = val(mf)
            gaps_lag.append((opt - vl) / opt); gaps_fill.append((opt - vf) / opt)
            exact_hits += int(abs(opt - vf) < 1e-6 * max(1, opt))
        out[ds] = {"n": len(gaps_fill), "gap_lagrangian_mean": float(np.mean(gaps_lag)), "gap_fill_mean": float(np.mean(gaps_fill)),
                   "gap_fill_p95": float(np.percentile(gaps_fill, 95)), "exact_rate": exact_hits / max(1, len(gaps_fill))}
        print(ds, out[ds], flush=True)
    return out


# ---------------------------------------------------------------- bootstrap
def bootstrap(reader, pairs, dss, B=2000, seed=0):
    rng = np.random.default_rng(seed); out = {}
    for ds in dss:
        for a, b in pairs:
            pa, pb = f"results/preds/{reader}/{ds}/{a}.jsonl", f"results/preds/{reader}/{ds}/{b}.jsonl"
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            A = {json.loads(l)["id"]: json.loads(l)["em"] for l in open(pa)}
            Bm = {json.loads(l)["id"]: json.loads(l)["em"] for l in open(pb)}
            ids = sorted(set(A) & set(Bm))
            d = np.array([A[i] - Bm[i] for i in ids])
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
            out[f"{ds}:{a}-vs-{b}"] = {"diff": 100 * float(d.mean()), "ci_lo": 100 * float(np.percentile(bs, 2.5)),
                                       "ci_hi": 100 * float(np.percentile(bs, 97.5)), "p_le0": float((bs <= 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="vc4")
    ap.add_argument("--datasets", default="nq,hotpotqa,musique,2wiki")
    ap.add_argument("--what", default="calib,gap,boot")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--reader", default="qwen7b")
    a = ap.parse_args()
    dss = a.datasets.split(",")
    res = json.load(open("results/analysis.json")) if os.path.exists("results/analysis.json") else {}
    if "calib" in a.what:
        res["calibration"] = calibration(a.name, dss); print(res["calibration"], flush=True)
    if "gap" in a.what:
        res["duality_gap"] = duality_gap(a.name, dss)
    if "boot" in a.what and a.pairs:
        pairs = [tuple(x.split(":")) for x in a.pairs.split(",")]
        res.setdefault("bootstrap", {}).update({f"{a.reader}|{k}": v for k, v in bootstrap(a.reader, pairs, dss).items()})
    json.dump(res, open("results/analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
