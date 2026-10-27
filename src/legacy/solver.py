"""Exact fidelity-constrained selection = maximum-weight closure (solved by one s-t min cut),
parametric budget search, and conformal risk control (CRC) calibration."""
import math
import networkx as nx
import numpy as np


def max_weight_closure(weights, deps):
    """weights: list of floats per unit (unit 0 ignored); deps: (u, v) meaning keep u => keep v.
    Returns bool list: the maximum-weight closed set (Picard 1976)."""
    n = len(weights)
    G = nx.DiGraph()
    G.add_node("s"); G.add_node("t")
    for i in range(1, n):
        w = weights[i]
        if w > 0:
            G.add_edge("s", i, capacity=w)
        elif w < 0:
            G.add_edge(i, "t", capacity=-w)
        else:
            G.add_node(i)
    for u, v in deps:
        G.add_edge(u, v, capacity=float("inf"))
    _, (S, _) = nx.minimum_cut(G, "s", "t")
    return [i in S for i in range(n)]


def max_weight_closure_fast(weights, deps):
    """Same optimum; exploits that dependency components are tiny (chains inside one passage + title).
    Enumerates closures exactly per connected component when small, falls back to min-cut otherwise."""
    n = len(weights)
    adj = {i: set() for i in range(1, n)}
    for u, v in deps:
        adj[u].add(v); adj[v].add(u)
    seen, kept = set(), [False] * n
    for s in range(1, n):
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            x = stack.pop(); comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y); stack.append(y)
        cd = [(u, v) for u, v in deps if u in comp]
        if len(comp) == 1:
            kept[comp[0]] = weights[comp[0]] > 0
            continue
        sub_w = [0.0] + [weights[c] for c in comp]
        m = {c: i + 1 for i, c in enumerate(comp)}
        sol = max_weight_closure(sub_w, [(m[u], m[v]) for u, v in cd])
        for c in comp:
            kept[c] = sol[m[c]]
    return kept


def solve(p, lengths, deps, lam, mean_len=25.0):
    w = [0.0] + [p[i] - lam * lengths[i] / mean_len for i in range(1, len(p))]
    return max_weight_closure_fast(w, deps)


def cost(kept, lengths):
    return sum(l for k, l in zip(kept, lengths) if k)


def solve_budget(p, lengths, deps, budget, mean_len=25.0, iters=30):
    """Parametric search on the token price lambda so that the closure fits the token budget."""
    lo, hi = 0.0, 50.0
    best = solve(p, lengths, deps, hi, mean_len)
    if cost(best, lengths) > budget:
        return best
    for _ in range(iters):
        mid = (lo + hi) / 2
        k = solve(p, lengths, deps, mid, mean_len)
        if cost(k, lengths) <= budget:
            best, hi = k, mid
        else:
            lo = mid
    return fill_budget(best, p, lengths, deps, budget)


def _closure_of(u, kept, out_adj):
    """units that must be added together with u to keep the selection closed under the constraints"""
    need, stack = {u}, [u]
    while stack:
        x = stack.pop()
        for y in out_adj.get(x, ()):
            if not kept[y] and y not in need:
                need.add(y); stack.append(y)
    return need


def fill_budget(kept, p, lengths, deps, budget):
    """Exact parametric closures are coarse in lambda; spend the residual budget greedily on the
    closed unit groups with the best necessity per token (every step keeps the set closed)."""
    kept = list(kept)
    out_adj = {}
    for u, v in deps:
        out_adj.setdefault(u, []).append(v)
    used = cost(kept, lengths)
    while True:
        best, best_ratio = None, 0.0
        for u in range(1, len(p)):
            if kept[u] or lengths[u] == 0:
                continue
            grp = _closure_of(u, kept, out_adj)
            c = sum(lengths[x] for x in grp)
            if used + c > budget or c == 0:
                continue
            ratio = sum(p[x] for x in grp) / c
            if ratio > best_ratio:
                best, best_ratio = grp, ratio
        if best is None or best_ratio < 1e-4:
            return kept
        for x in best:
            kept[x] = True
        used += sum(lengths[x] for x in best)


def miss_rate(kept, gold):
    g = [i for i, y in enumerate(gold) if y]
    if not g:
        return 0.0
    return sum(1 for i in g if not kept[i]) / len(g)


def crc_lambda(loss_curves, lams, alpha, B=1.0):
    """loss_curves: n x |lams| (non-decreasing in lambda). Return the largest lambda with
    n/(n+1) * Rhat(lambda) + B/(n+1) <= alpha  (Angelopoulos et al., 2024)."""
    L = np.asarray(loss_curves)
    n = L.shape[0]
    rhat = L.mean(0)
    ok = (n / (n + 1)) * rhat + B / (n + 1) <= alpha
    best = None
    for j, lam in enumerate(lams):
        if ok[: j + 1].all():
            best = lam
    return best if best is not None else lams[0]


# ---------------- neural CRF MAP: max sum_u w_u m_u + sum_{u<v} c_uv m_u m_v  (c >= 0) ----------------
def crf_map(w, C, deps, scale=1e4, P=None):
    """Exact MAP of a submodular pairwise energy by one s-t min cut (scipy, integer capacities).
    -c m_u m_v = -c m_u + c m_u (1 - m_v): add c to w_u and an arc u->v of capacity c.
    Hard anchors (u->v in deps) are arcs of (effectively) infinite capacity.
    Directed dependency penalties P[u, v] * m_u * (1 - m_v) (keep u without the context v it needs) are arcs u->v."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow
    n = len(w)
    S, T = n, n + 1
    wp = np.array(w, dtype=np.float64)
    rows, cols, caps = [], [], []
    iu, ju = np.nonzero(np.triu(C, 1))
    for u, v in zip(iu, ju):
        c = C[u, v]
        wp[u] += c
        rows.append(u); cols.append(v); caps.append(c)
    if P is not None:
        iu, ju = np.nonzero(P > 0)
        for u, v in zip(iu, ju):
            rows.append(u); cols.append(v); caps.append(P[u, v])
    big = float(np.abs(wp).sum() + C.sum() + (P.sum() if P is not None else 0.0) + 1.0)
    for u, v in deps:
        rows.append(u); cols.append(v); caps.append(big)
    for u in range(1, n):
        if wp[u] > 0:
            rows.append(S); cols.append(u); caps.append(wp[u])
        elif wp[u] < 0:
            rows.append(u); cols.append(T); caps.append(-wp[u])
    caps = np.maximum(np.round(np.array(caps) * scale), 0).astype(np.int64)
    # merge duplicate arcs
    G = csr_matrix((caps, (rows, cols)), shape=(n + 2, n + 2))
    G.sum_duplicates()
    res = maximum_flow(G.astype(np.int32) if G.max() < 2**31 else G, S, T)
    R = (G - res.flow).tocsr()
    # source side of the min cut = vertices reachable from S in the residual graph
    seen = np.zeros(n + 2, bool); seen[S] = True; stack = [S]
    while stack:
        x = stack.pop()
        lo, hi = R.indptr[x], R.indptr[x + 1]
        for y, cap in zip(R.indices[lo:hi], R.data[lo:hi]):
            if cap > 0 and not seen[y]:
                seen[y] = True; stack.append(y)
    return [bool(seen[i]) and i > 0 for i in range(n)]


def crf_solve(p, C, lengths, deps, lam, mean_len=25.0, P=None):
    w = [0.0] + [p[i] - lam * lengths[i] / mean_len for i in range(1, len(p))]
    return crf_map(w, C, deps, P=P)


def crf_value(kept, p, C):
    import numpy as np
    m = np.array(kept, dtype=float)
    return float(np.dot(p, m) + 0.5 * m @ C @ m)


def crf_solve_budget(p, C, lengths, deps, budget, mean_len=25.0, iters=25, P=None):
    lo, hi = 0.0, 50.0
    best = crf_solve(p, C, lengths, deps, hi, mean_len, P)
    if cost(best, lengths) > budget:
        return best
    for _ in range(iters):
        mid = (lo + hi) / 2
        k = crf_solve(p, C, lengths, deps, mid, mean_len, P)
        if cost(k, lengths) <= budget:
            best, hi = k, mid
        else:
            lo = mid
    # residual fill: marginal CRF gain per token, keeping anchors closed
    import numpy as np
    kept = list(best)
    out_adj = {}
    for u, v in deps:
        out_adj.setdefault(u, []).append(v)
    used = cost(kept, lengths)
    m = np.array(kept, dtype=float)
    while True:
        bestg, br = None, 0.0
        for u in range(1, len(p)):
            if kept[u] or lengths[u] == 0:
                continue
            grp = _closure_of(u, kept, out_adj)
            c = sum(lengths[x] for x in grp)
            if used + c > budget:
                continue
            gain = sum(p[x] + float(C[x] @ m) for x in grp)
            if P is not None:
                g = list(grp)
                after = m.copy(); after[g] = 1.0
                gain -= float(P[g].dot(1.0 - after).sum())      # new units whose context is still missing
                gain += float(m.dot(P[:, g]).sum())               # kept units whose missing context is now restored
            if gain / c > br:
                bestg, br = grp, gain / c
        if bestg is None or br < 1e-4:
            return kept
        for x in bestg:
            kept[x] = True; m[x] = 1.0
        used += sum(lengths[x] for x in bestg)
