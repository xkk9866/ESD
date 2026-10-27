"""Train CAET end-to-end with a single objective: cost-sensitive keep loss + on-policy self-verification losses."""
import argparse, json, math, os, random, time
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from caet import CAET, Featurizer, dependency_list


class JsonlFeat(Dataset):
    def __init__(self, path, feat, n=None, anchors_only=False, dep_file=None):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")][:n]
        self.feat = feat
        self.anchors_only = anchors_only
        self.deps = {}
        if dep_file:
            for l in open(dep_file, encoding="utf-8"):
                d = json.loads(l)
                self.deps[d["id"]] = d["pairs"]
            self.rows = [r for r in self.rows if r["id"] in self.deps]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ex = self.rows[i]
        f = self.feat(ex)
        f["deps"] = dependency_list(ex, f["units"], self.anchors_only)
        idx = {(u["pid"], u["sid"]): i for i, u in enumerate(f["units"])}
        # LLM-distilled contextual dependencies: unit a cannot be read correctly without unit b (label 1/2)
        f["dep_pairs"] = [(idx[(pi, a)], idx[(pi, b)], 1.0 if lab in (1, 2) else 0.0)
                          for pi, a, b, lab in self.deps.get(ex["id"], []) if (pi, a) in idx and (pi, b) in idx]
        return f


def closure(kept, deps):
    changed = True
    while changed:
        changed = False
        for u, v in deps:
            if kept[u] and not kept[v]:
                kept[v] = True; changed = True
    return kept


def sample_selection(p, units, deps, rng, ablate_closure=False):
    t = rng.uniform(0.15, 0.7)
    kept = [False] * len(units)
    for i, u in enumerate(units):
        if u["kind"] != "sent":
            continue
        k = p[i] > t
        if u["gold"] and k and rng.random() < 0.3:
            k = False
        if not u["gold"] and not k and rng.random() < 0.05:
            k = True
        kept[i] = k
    if not ablate_closure:
        kept = closure(kept, deps)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="models/bge-reranker-v2-m3")
    ap.add_argument("--train", default="data/proc/train.jsonl")
    ap.add_argument("--out", default="ckpt/caet")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--n_global", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eta", type=float, default=0.5, help="token-price multiplier (Lagrangian of budget constraint)")
    ap.add_argument("--pos_w", type=float, default=2.0)
    ap.add_argument("--mu_omit", type=float, default=0.5)
    ap.add_argument("--mu_suff", type=float, default=0.2)
    ap.add_argument("--no_verify", action="store_true")
    ap.add_argument("--no_relbias", action="store_true")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--crf", action="store_true", help="neural CRF couplings trained by pseudo-likelihood")
    ap.add_argument("--structural", action="store_true", help="structural relations only (no lexical rules)")
    ap.add_argument("--dep_file", default=None, help="LLM-distilled dependency labels")
    ap.add_argument("--mu_dep", type=float, default=1.0)
    ap.add_argument("--init", default=None, help="warm-start state dict")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    torch.manual_seed(0); rng = random.Random(0)

    tok = AutoTokenizer.from_pretrained(args.backbone)
    ds = JsonlFeat(args.train, Featurizer(tok, max_len=args.max_len, structural=args.structural), args.n, anchors_only=args.structural, dep_file=args.dep_file)
    dl = DataLoader(ds, batch_size=None, shuffle=True, num_workers=6, prefetch_factor=4, persistent_workers=True)
    model = CAET(args.backbone, n_global=args.n_global)
    if args.init:
        print(model.load_state_dict(torch.load(os.path.join(args.init, "model.pt"), map_location="cpu"), strict=False), flush=True)
    model = model.cuda()
    model.enc.embeddings.word_embeddings.weight.requires_grad_(False)  # frozen vocab table (memory)
    if args.no_relbias:
        model.rel_bias.requires_grad_(False)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": args.lr},
                             {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
    total = len(ds) * args.epochs // args.accum
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    mean_len = 25.0
    model.train(); step = 0; t0 = time.time(); log = {"keep": 0, "omit": 0, "suff": 0, "n": 0}
    for ep in range(args.epochs):
        for it, f in enumerate(dl):
            units = f["units"]
            sent_idx = [i for i, u in enumerate(units) if u["kind"] == "sent"]
            if not sent_idx:
                continue
            y = torch.tensor([units[i]["gold"] for i in sent_idx], dtype=torch.float, device="cuda")
            ln = torch.tensor([units[i]["nwords"] for i in sent_idx], dtype=torch.float, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                keep, cache = model.read(f)
            kl = keep.float()[sent_idx]
            # cost-sensitive BCE = Lagrangian relaxation of "max preserved evidence s.t. token budget"
            w = torch.where(y > 0, torch.full_like(y, args.pos_w), 1.0 + args.eta * ln / mean_len)
            l_keep = (F.binary_cross_entropy_with_logits(kl, y, reduction="none") * w).sum() / w.sum()
            loss = l_keep
            if args.crf:
                # pseudo-likelihood of the pairwise CRF: logit(y_u | y_-u) = a_u + sum_v c_uv * y_v
                C = model.couplings(cache)[sent_idx][:, sent_idx]
                yc = y.clone()
                flip = torch.rand_like(yc) < 0.1          # context noise -> robustness to imperfect partners
                yc = torch.where(flip, 1 - yc, yc)
                z = kl + C @ yc
                l_pl = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() / w.sum()
                loss = loss + l_pl
                log["omit"] += 0.0
            if args.dep_file and f["dep_pairs"]:
                # plain (unweighted) BCE so that D_uv is a calibrated probability: it enters the solver as an expected loss
                DL = model.dependency_logits(cache)
                ia = torch.tensor([x[0] for x in f["dep_pairs"]], device="cuda")
                ib = torch.tensor([x[1] for x in f["dep_pairs"]], device="cuda")
                yd = torch.tensor([x[2] for x in f["dep_pairs"]], device="cuda")
                l_dep = F.binary_cross_entropy_with_logits(DL[ia, ib], yd)
                loss = loss + args.mu_dep * l_dep
                log["dep"] = log.get("dep", 0.0) + l_dep.item()
            if not args.no_verify:
                p = torch.sigmoid(keep.detach().float()).tolist()
                kept = sample_selection(p, units, f["deps"], rng)
                kept_t = torch.tensor(kept, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    omit, suff = model.verify(cache, kept_t)
                drop_idx = [i for i in sent_idx if not kept[i]]
                l_omit = torch.zeros((), device="cuda")
                if drop_idx:
                    yo = torch.tensor([units[i]["gold"] for i in drop_idx], dtype=torch.float, device="cuda")
                    wo = torch.where(yo > 0, torch.full_like(yo, args.pos_w), torch.ones_like(yo))
                    l_omit = (F.binary_cross_entropy_with_logits(omit.float()[drop_idx], yo, reduction="none") * wo).sum() / wo.sum()
                ys = torch.tensor(float(all(kept[i] for i in sent_idx if units[i]["gold"])), device="cuda")
                l_suff = F.binary_cross_entropy_with_logits(suff.float(), ys)
                loss = loss + args.mu_omit * l_omit + args.mu_suff * l_suff
                log["omit"] += l_omit.item(); log["suff"] += l_suff.item()
            (loss / args.accum).backward()
            log["keep"] += l_keep.item(); log["n"] += 1
            if (it + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % args.log_every == 0:
                    n = log["n"]
                    print(f"ep{ep} step {step}/{total} keep {log['keep']/n:.4f} omit {log['omit']/n:.4f} "
                          f"suff {log['suff']/n:.4f} dep {log.get('dep', 0)/n:.4f} {(time.time()-t0)/60:.1f}min", flush=True)
                    log = {"keep": 0, "omit": 0, "suff": 0, "n": 0}
    torch.save({k: v for k, v in model.state_dict().items()}, os.path.join(args.out, "model.pt"))
    tok.save_pretrained(args.out)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
