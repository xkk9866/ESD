"""Posterior Set Decoding (PSD): read the trained set-decoding policy as a generative model of sufficient sets and
output the Bayes decision under it, instead of the single greedy trajectory.

The policy pi(a | q, C) of ESD defines, by ancestral sampling until END, a distribution P(S | q) over complete evidence
sets (the state is rendered as a set, so orderings that reach the same set share every state). Greedy decoding returns
one high-probability trajectory and ignores how much of P lies elsewhere. Measured on our chains, the END probability
at the halting state is ~0.97-1.0 even where the policy was unsure which unit to take (NQ: 30% of the chosen units have
pi < 0.5, and the decoded set contains the gold set for only 45% of the questions). Halting is therefore blind to the
uncertainty of selection.

PSD draws N trajectories from pi, and returns the smallest set C that the model believes, more likely than not,
contains a sufficient set:
        C* = argmin_C |C|   s.t.   P(S subset-of C | q) >= 1/2,    P estimated by the fraction of sampled sets inside C.
The level 1/2 is the majority decision ("C is sufficient" vs "C is not"), the same semantics as END winning the argmax.
When the policy is confident, every sample is the same set and C* equals it; when it is split between alternatives,
C* is the union of the alternatives that together carry the majority of the mass. The length of the compression is thus
determined by the uncertainty of the model -- one network, one distribution, one decision rule.

Outputs (results/compressed/<ds>/):  <name> (credible set at 1/2), <name>_lv{l} (other levels, for the curve),
  <name>_f1 (minimum-Bayes-risk set under expected F1), <name>_mode (most frequent sampled set: sampling without
  aggregation), <name>_gx (greedy ESD chain padded with the policy's empty-state ranking to C*'s size: same length,
  without the posterior), <name>_marg (empty-state ranking at C*'s size)."""
import argparse, itertools, json, os, random, time
from collections import Counter
import numpy as np, torch
from esd import Net, units, state, END
from baselines import load, render_kept


class PosteriorDecoder:
    def __init__(self, path, n_samples=16, max_steps=12, max_len=512, bs=128):
        self.net = Net(path, max_len); self.net.m.eval()
        self.N, self.max_steps, self.bs = n_samples, max_steps, bs

    def _dist(self, q, txt, C, cache):
        key = frozenset(C)
        if key not in cache:
            pool = [j for j in range(len(txt)) if j not in key]
            z = self.net.logits([state(q, txt, sorted(key))] * (len(pool) + 1), [txt[j] for j in pool] + [END], bs=self.bs)
            if not key:
                z[-1] = -np.inf
            p = np.exp(z - z.max()); p /= p.sum()
            cache[key] = (pool, p, z)
        return cache[key]

    @torch.no_grad()
    def decode(self, ex, seed):
        us = units(ex); txt = [u[2] for u in us]; q = ex["question"]
        rng = np.random.default_rng(seed); cache = {}
        cap = min(self.max_steps, len(us))
        # greedy trajectory (ESD) -- shares the cache with the samples
        C, greedy = [], []
        while len(C) < cap:
            pool, p, _ = self._dist(q, txt, C, cache)
            k = int(np.argmax(p))
            if k == len(pool):
                break
            C.append(pool[k]); greedy.append(pool[k])
        samples = []
        for _ in range(self.N):
            C = []
            while len(C) < cap:
                pool, p, _ = self._dist(q, txt, C, cache)
                k = int(rng.choice(len(p), p=p))
                if k == len(pool):
                    break
                C.append(pool[k])
            samples.append(tuple(sorted(C)))
        z0 = self._dist(q, txt, [], cache)[2][:-1].tolist()
        return {"z0": z0, "greedy": greedy, "samples": samples, "states": len(cache)}


# ------------------------------------------------------------------ decision rules over the sampled sets
def credible_set(samples, level=0.5):
    """smallest union of sampled sets whose contained mass reaches `level` (exact over distinct sets, ties -> more mass)"""
    cnt = Counter(samples); D = list(cnt); N = len(samples)
    allu = sorted({u for s in D for u in s}); bit = {u: 1 << i for i, u in enumerate(allu)}
    mask = [sum(bit[u] for u in s) for s in D]; mass = [cnt[s] / N for s in D]
    need = level - 1e-9
    def contained(m):
        return sum(w for mk, w in zip(mask, mass) if mk & m == mk)
    best = None
    if len(D) <= 14:
        for r in range(1, len(D) + 1):
            for comb in itertools.combinations(range(len(D)), r):
                m = 0
                for i in comb:
                    m |= mask[i]
                w = contained(m)
                if w >= need:
                    key = (bin(m).count("1"), -w)
                    if best is None or key < best[0]:
                        best = (key, m)
    else:  # greedy: add the distinct set that adds the fewest new units per unit of mass
        m, w = 0, 0.0
        order = sorted(range(len(D)), key=lambda i: -mass[i])
        while w < need:
            i = min((i for i in order if mask[i] & m != mask[i]), key=lambda i: (bin(mask[i] & ~m).count("1") / mass[i]))
            m |= mask[i]; w = contained(m)
        best = ((bin(m).count("1"), -w), m)
    m = best[1]
    return [u for u in allu if m & bit[u]]


def mbr_f1(samples):
    """the set maximising the expected F1 against S ~ P; candidates: prefixes of the posterior inclusion ranking"""
    N = len(samples); S = [set(s) for s in samples]
    marg = Counter(u for s in S for u in s)
    order = [u for u, _ in marg.most_common()]
    best, bestv = None, -1
    for k in range(1, len(order) + 1):
        C = set(order[:k])
        v = np.mean([2 * len(C & s) / (len(C) + len(s)) if s else 0.0 for s in S])
        if v > bestv + 1e-12:
            best, bestv = sorted(C), v
    return best


def mode_set(samples):
    return list(Counter(samples).most_common(1)[0][0])


def pad(base, z0, k):
    out = list(base)[:k]
    for j in np.argsort(-np.asarray(z0)):
        if len(out) >= k:
            break
        if int(j) not in out:
            out.append(int(j))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/esd")
    ap.add_argument("--name", default="psd")
    ap.add_argument("--datasets", default="nq,hotpotqa,musique,2wiki")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--levels", default="0.3,0.7,0.9")
    a = ap.parse_args()
    dec = PosteriorDecoder(a.ckpt, a.samples, a.max_steps)
    levels = [float(x) for x in a.levels.split(",") if x]
    for ds in a.datasets.split(","):
        os.makedirs(f"results/chains/{ds}", exist_ok=True); os.makedirs(f"results/compressed/{ds}", exist_ok=True)
        exs = load(ds, a.n); path = f"results/chains/{ds}/{a.name}.jsonl"; done = {}
        if os.path.exists(path):
            for l in open(path, encoding="utf-8"):
                try:
                    r = json.loads(l); done[r["id"]] = r
                except json.JSONDecodeError:
                    pass
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in done.values())
        with open(path, "a", encoding="utf-8") as f:
            for n, e in enumerate(exs):
                if e["id"] in done:
                    continue
                t0 = time.time(); r = dec.decode(e, seed=n); r["time"] = time.time() - t0; r["id"] = e["id"]
                done[e["id"]] = r; f.write(json.dumps(r) + "\n"); f.flush()
                if n % 100 == 0:
                    print(ds, n, "states", r["states"], "%.2fs" % r["time"], flush=True)
        recs = [done[e["id"]] for e in exs]
        rules = {a.name: lambda r: credible_set(r["samples"], 0.5),
                 f"{a.name}_f1": lambda r: mbr_f1(r["samples"]),
                 f"{a.name}_mode": lambda r: mode_set(r["samples"])}
        for lv in levels:
            rules[f"{a.name}_lv{lv}"] = (lambda lv: lambda r: credible_set(r["samples"], lv))(lv)
        sel = {k: [fn(r) for r in recs] for k, fn in rules.items()}
        size = [len(s) for s in sel[a.name]]
        sel[f"{a.name}_gx"] = [pad(r["greedy"], r["z0"], k) for r, k in zip(recs, size)]
        sel[f"{a.name}_marg"] = [pad([], r["z0"], k) for r, k in zip(recs, size)]
        for k, S in sel.items():
            with open(f"results/compressed/{ds}/{k}.jsonl", "w", encoding="utf-8") as f:
                for e, r, s in zip(exs, recs, S):
                    us = units(e); kept = [(us[i][0], us[i][1]) for i in s]
                    f.write(json.dumps({"id": e["id"], "context": render_kept(e, kept), "kept": kept, "time": r["time"]}) + "\n")
        print(a.name, ds, " ".join(f"{k}:{np.mean([len(s) for s in S]):.2f}" for k, S in sel.items()),
              "greedy %.2f states %.1f sec %.2f" % (np.mean([len(r["greedy"]) for r in recs]), np.mean([r["states"] for r in recs]),
                                                    np.mean([r["time"] for r in recs])), flush=True)


if __name__ == "__main__":
    main()
