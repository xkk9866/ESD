"""VeriComp inference: read -> exact constrained selection -> self-verify/repair -> certified fallback.
Writes compressed evidence for every dataset and variant into results/compressed/<ds>/<method>.jsonl"""
import argparse, json, os, pickle, time
import numpy as np, torch
from transformers import AutoTokenizer
from caet import CAET, Featurizer, dependency_list
from solver import solve, solve_budget, miss_rate, crc_lambda, cost, crf_solve, crf_solve_budget

DATASETS = ["nq", "hotpotqa", "musique", "2wiki"]
LAMS = list(np.round(np.concatenate([np.linspace(0.0, 1.0, 41), np.linspace(1.05, 3.0, 40)]), 4))
# price grid: log-spaced, because calibrated necessities of irrelevant units are tiny
PLAMS = [0.0] + list(np.logspace(-4, np.log10(5.0), 60))


def render(ex, units, kept, with_title=True):
    by_p = {}
    for i, u in enumerate(units):
        if kept[i] and u["kind"] == "sent":
            by_p.setdefault(u["pid"], []).append(ex["passages"][u["pid"]]["sents"][u["sid"]])
    out = []
    for pi in sorted(by_p):
        body = " ".join(by_p[pi])
        out.append(f"{ex['passages'][pi]['title']}: {body}" if with_title else body)
    return "\n".join(out)


def full_kept(units):
    return [i > 0 for i in range(len(units))]


class Runner:
    def __init__(self, ckpt, backbone, n_global, max_len=4096, crf=False, dep=False, cpl=True, bayes=False):
        self.crf, self.dep, self.cpl, self.bayes = crf, dep, cpl, bayes
        targs = json.load(open(os.path.join(ckpt, "args.json"))) if os.path.exists(os.path.join(ckpt, "args.json")) else {}
        self.pos_w, self.eta = targs.get("pos_w", 2.0), targs.get("eta", 0.5)
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = CAET(backbone, n_global=n_global, grad_ckpt=False)
        self.model.load_state_dict(torch.load(os.path.join(ckpt, "model.pt"), map_location="cpu"), strict=False)
        self.model = self.model.cuda().eval().to(torch.bfloat16)
        self.feat = Featurizer(self.tok, max_len=max_len, structural=crf)

    @torch.no_grad()
    def read(self, ex):
        f = self.feat(ex)
        keep, cache = self.model.read(f)
        p = torch.sigmoid(keep.float()).tolist()
        units = f["units"]
        p = [0.0 if u["kind"] != "sent" else x for u, x in zip(units, p)]  # titles carry cost only (anchors)
        lengths = [0] + [u["nwords"] for u in units[1:]]
        cache["p_raw"] = list(p)
        if self.bayes:
            # exact inversion of the cost-sensitive BCE optimum p = w+ q / (w+ q + w- (1-q)), w- = 1 + eta * len / 25
            q = []
            for x, l in zip(p, lengths):
                wn = 1.0 + self.eta * l / 25.0
                q.append(0.0 if x <= 0 else wn * x / (self.pos_w * (1 - x) + wn * x))
            p = q
        deps = dependency_list(ex, units, anchors_only=self.crf)
        U = len(units)
        cache["C"] = (self.model.couplings(cache).float().cpu().numpy() if (self.crf and self.cpl) else np.zeros((U, U)))
        if self.dep:
            D = self.model.dependencies(cache).float().cpu().numpy()
            pid = np.array([u["pid"] for u in units]); sent = np.array([u["kind"] == "sent" for u in units])
            same = (pid[:, None] == pid[None, :]) & sent[:, None] & sent[None, :]      # structural scope: same passage
            cache["D"] = D * same
            cache["P"] = np.asarray(p)[:, None] * cache["D"]        # expected loss of keeping u without its context v
        return f, p, units, lengths, deps, cache

    @torch.no_grad()
    def verify(self, cache, kept):
        omit, suff = self.model.verify(cache, torch.tensor(kept, device="cuda"))
        return torch.sigmoid(omit.float()).tolist(), torch.sigmoid(suff.float()).item()


def passage_miss(kept, units):
    """fraction of gold passages whose gold evidence is entirely dropped (monotone in the kept set)"""
    gp = {}
    for i, u in enumerate(units):
        if u["kind"] == "sent" and u["gold"]:
            gp.setdefault(u["pid"], []).append(kept[i])
    if not gp:
        return 0.0
    return sum(1 for v in gp.values() if not any(v)) / len(gp)


def _solve(R, p, lengths, deps, cache, lam):
    if R.crf or R.dep:
        return crf_solve(p, cache["C"], lengths, deps, lam, P=cache.get("P"))
    return solve(p, lengths, deps, lam)


def _solve_budget(R, p, lengths, deps, cache, B):
    if R.crf or R.dep:
        return crf_solve_budget(p, cache["C"], lengths, deps, B, P=cache.get("P"))
    return solve_budget(p, lengths, deps, B)


def gold_of(units):
    return [u["gold"] if u["kind"] == "sent" else 0 for u in units]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/caet")
    ap.add_argument("--backbone", default="models/bge-reranker-v2-m3")
    ap.add_argument("--n_global", type=int, default=4)
    ap.add_argument("--name", default="vericomp")
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3")
    ap.add_argument("--main_alpha", type=float, default=0.1)
    ap.add_argument("--tau_s", type=float, default=0.15)
    ap.add_argument("--repair_iters", type=int, default=2)
    ap.add_argument("--ratios", default="2,4,8")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--crf", action="store_true")
    ap.add_argument("--dep", action="store_true", help="directed contextual-dependency penalties from the dependency head")
    ap.add_argument("--no_cpl", action="store_true", help="drop the symmetric co-selection couplings")
    ap.add_argument("--bayes", action="store_true", help="invert the cost-sensitive BCE to posterior necessities")
    ap.add_argument("--dump", action="store_true", help="dump per-unit probabilities and dependencies for analysis")
    ap.add_argument("--crc_loss", default="passage", choices=["passage", "sentence"])
    ap.add_argument("--max_restore", type=int, default=3)
    ap.add_argument("--price_ratios", default="", help="corpus-level price mode: average budgets total/r, lambda set on calib")
    ap.add_argument("--only_price", action="store_true")
    args = ap.parse_args()
    R = Runner(args.ckpt, args.backbone, args.n_global, crf=args.crf, dep=args.dep, cpl=not args.no_cpl, bayes=args.bayes)
    os.makedirs("results/dumps", exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]
    ratios = [int(r) for r in args.ratios.split(",")]
    os.makedirs("results/calib", exist_ok=True)
    for ds in args.datasets.split(","):
        out_dir = f"results/compressed/{ds}"; os.makedirs(out_dir, exist_ok=True)
        # ---------- 1) conformal risk control on the calibration split ----------
        cal = [json.loads(l) for l in open(f"data/proc/calib_{ds}.jsonl", encoding="utf-8")]
        curves, costs, costs_nc, totals = [], [], [], []
        for ex in cal:
            f, p, units, lengths, deps, cc = R.read(ex)
            g = gold_of(units)
            sols = [_solve(R, p, lengths, deps, cc, lam) for lam in LAMS]
            if args.crc_loss == "passage":
                curves.append([passage_miss(k, units) for k in sols])
            else:
                curves.append([miss_rate(k, g) for k in sols])
            if args.price_ratios:
                costs.append([cost(_solve(R, p, lengths, deps, cc, lam), lengths) for lam in PLAMS])
                if R.crf or R.dep:
                    costs_nc.append([cost(solve(p, lengths, deps, lam), lengths) for lam in PLAMS])
            totals.append(cost(full_kept(units), lengths))
        price_ratios = [int(r) for r in args.price_ratios.split(",")] if args.price_ratios else []

        def price_for(cst, r):
            """smallest token price whose average calibration cost meets the average budget total/r"""
            target = np.mean(totals) / r
            mc = np.mean(cst, 0)
            ok = [lam for lam, c in zip(PLAMS, mc) if c <= target]
            return min(ok) if ok else PLAMS[-1]
        lam_price = {r: price_for(costs, r) for r in price_ratios}
        lam_price_nc = {r: price_for(costs_nc, r) for r in price_ratios} if costs_nc else {}
        print(ds, "price lambdas", lam_price, lam_price_nc, flush=True)
        lam_hat = {a: crc_lambda(curves, LAMS, a) for a in alphas}
        json.dump({"lam_hat": lam_hat, "rhat": np.mean(curves, 0).tolist(), "lams": LAMS},
                  open(f"results/calib/{args.name}_{ds}.json", "w"))
        print(ds, "lambda_hat", lam_hat, flush=True)
        # ---------- 2) test ----------
        test = [json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")][: args.n_test]
        dumpf = open(f"results/dumps/{args.name}_{ds}.jsonl", "w") if args.dump else None
        writers = {}

        def emit(method, ex, units, kept, t, extra=None, with_title=True):
            if method not in writers:
                writers[method] = open(f"{out_dir}/{method}.jsonl", "w", encoding="utf-8")
            rec = {"id": ex["id"], "context": render(ex, units, kept, with_title),
                   "kept": [[u["pid"], u["sid"]] for i, u in enumerate(units) if kept[i] and u["kind"] == "sent"],
                   "time": t}
            if extra:
                rec.update(extra)
            writers[method].write(json.dumps(rec, ensure_ascii=False) + "\n")

        for n_ex, ex in enumerate(test):
            torch.cuda.synchronize(); t0 = time.time()
            f, p, units, lengths, deps, cache = R.read(ex)
            torch.cuda.synchronize(); t_read = time.time() - t0
            total = cost(full_kept(units), lengths)
            if args.dump:
                rec = {"id": ex["id"], "u": [[u["pid"], u["sid"], round(cache["p_raw"][i], 5), round(p[i], 5), u["gold"], lengths[i]]
                                             for i, u in enumerate(units) if u["kind"] == "sent"]}
                if R.dep:
                    nz = np.argwhere(cache["D"] > 0.05)
                    rec["D"] = [[units[a]["pid"], units[a]["sid"], units[b]["sid"], round(float(cache["D"][a, b]), 4)] for a, b in nz]
                dumpf.write(json.dumps(rec) + "\n")
            for rt in price_ratios:
                lam = lam_price[rt]
                kept = _solve(R, p, lengths, deps, cache, lam)
                r_, s_ = R.verify(cache, kept)
                emit(f"{args.name}_p{rt}", ex, units, kept, t_read, {"suff": s_, "pmiss": passage_miss(kept, units)})
                if R.crf or R.dep:
                    kn = solve(p, lengths, deps, lam_price_nc[rt])
                    emit(f"{args.name}-indep_p{rt}", ex, units, kn, t_read, {"pmiss": passage_miss(kn, units)})
            if args.only_price:
                if n_ex % 100 == 0:
                    print(ds, n_ex, flush=True)
                continue
            for a in alphas:
                t1 = time.time()
                lam = lam_hat[a]
                kept0 = _solve(R, p, lengths, deps, cache, lam)
                t_solve = time.time() - t1
                tag = f"{args.name}_a{a}"
                if a == args.main_alpha:
                    # ablation: no fidelity constraints (independent thresholding, no title anchoring)
                    kept_nc = [i > 0 and units[i]["kind"] == "sent" and p[i] - lam * lengths[i] / 25.0 > 0 for i in range(len(units))]
                    emit(f"{args.name}-noclosure", ex, units, kept_nc, t_read, with_title=False)
                    emit(f"{args.name}-noverify", ex, units, kept0, t_read + t_solve)
                # verify -> repair (monotone: weights only increase => closure only grows)
                t2 = time.time()
                q, kept, s = list(p), kept0, 1.0
                for _ in range(args.repair_iters):
                    r, s = R.verify(cache, kept)
                    newq = [max(q[i], r[i]) if (not kept[i] and units[i]["kind"] == "sent") else q[i] for i in range(len(q))]
                    if newq == q:
                        break
                    q = newq
                    kept = _solve(R, q, lengths, deps, cache, lam)
                r, s = R.verify(cache, kept)
                fb = False
                restored = 0
                while s < args.tau_s and restored < args.max_restore:
                    # graded fallback: restore the original passage of the riskiest dropped unit
                    cand = [(r[i], units[i]["pid"]) for i in range(1, len(units)) if units[i]["kind"] == "sent" and not kept[i]]
                    if not cand:
                        break
                    pid_r = max(cand)[1]
                    kept = [kept[i] or (i > 0 and units[i]["pid"] == pid_r) for i in range(len(units))]
                    restored += 1
                    r, s = R.verify(cache, kept)
                fb = s < args.tau_s
                if a == args.main_alpha:
                    emit(f"{args.name}-nofallback", ex, units, kept, t_read + t_solve + time.time() - t2, {"suff": s})
                final = full_kept(units) if fb else kept
                emit(tag, ex, units, final, t_read + t_solve + time.time() - t2, {"suff": s, "fallback": fb, "restored": restored})
                if a == args.main_alpha:
                    emit(args.name, ex, units, final, t_read + t_solve + time.time() - t2, {"suff": s, "fallback": fb})
            # budget mode (matched compression ratio): constrained knapsack via parametric closure + repair
            for rt in ratios:
                t1 = time.time()
                B = total / rt
                kept = _solve_budget(R, p, lengths, deps, cache, B)
                r, s = R.verify(cache, kept)
                q = [max(p[i], r[i]) if (not kept[i] and units[i]["kind"] == "sent") else p[i] for i in range(len(p))]
                kept = _solve_budget(R, q, lengths, deps, cache, B)
                emit(f"{args.name}_r{rt}", ex, units, kept, t_read + time.time() - t1)
                if R.crf:  # ablation: same network, couplings switched off (independent unaries)
                    emit(f"{args.name}-nocpl_r{rt}", ex, units, solve_budget(p, lengths, deps, B), t_read)
                    emit(f"{args.name}-norepair_r{rt}", ex, units, _solve_budget(R, p, lengths, deps, cache, B), t_read)
            if n_ex % 100 == 0:
                print(ds, n_ex, flush=True)
        for w in writers.values():
            w.close()


if __name__ == "__main__":
    main()
