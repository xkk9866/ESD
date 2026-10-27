"""Evidence Chain Decoding (ECD): compression as autoregressive decoding of an evidence set.

One cross-encoder models the chain rule of evidence selection,
    p(u in S* | q, C)      for any already-decoded chain C (C = empty -> the usual independent scorer),
trained on random prefixes of the gold evidence set (teacher forcing over set orderings).
Decoding appends the most necessary remaining unit given the chain and stops when the chain is more likely
sufficient than not, P(no remaining unit is needed | q, C) = prod_v (1 - p(v | q, C)) >= 1/2 (Bayes decision):
the *length* of the compression is decided by the model's own conditional estimate that the chain is already
sufficient, not by a ratio, top-k or threshold.
Every step's conditional probability is stored, so a corpus-level token price can truncate the same decoded
chain post hoc (budgeted operating points without re-decoding).
"""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

CTX_SEP = " <ctx> "


def units(ex):
    return [(pi, si, f"{p['title']}: {s}") for pi, p in enumerate(ex["passages"]) for si, s in enumerate(p["sents"])]


def query_with_chain(q, chain):
    return q if not chain else q + CTX_SEP + " | ".join(chain)


class Net:
    def __init__(self, path, max_len=384, train=False):
        self.tok = AutoTokenizer.from_pretrained(path)
        dt = torch.float32 if train else torch.bfloat16
        self.m = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=dt).cuda()
        self.max_len = max_len

    def enc(self, qs, us):
        return self.tok(qs, us, padding=True, truncation="longest_first", max_length=self.max_len, return_tensors="pt").to("cuda")

    @torch.no_grad()
    def logits(self, qs, us, bs=64):
        out = []
        for i in range(0, len(qs), bs):
            out += self.m(**self.enc(qs[i:i + bs], us[i:i + bs])).logits.view(-1).float().tolist()
        return np.asarray(out)


# ------------------------------------------------------------------ training: teacher forcing over gold-set orderings
def hard_negatives(a):
    """step-0 scores of the initial network on the training pool, so that the chain may contain the distractors
    the decoder itself is most likely to pick (training contexts resemble decoding contexts)"""
    cache = a.out + "_hardneg.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    net = Net(a.backbone); net.m.eval(); H = {}
    for n, l in enumerate(open(a.train, encoding="utf-8")):
        e = json.loads(l); us = units(e)
        z = net.logits([e["question"]] * len(us), [u[2] for u in us], bs=128)
        H[e["id"]] = [int(j) for j in np.argsort(-z) if not e["passages"][us[j][0]]["gold"][us[j][1]]][:8]
        if n % 2000 == 0:
            print("hardneg", n, flush=True)
    json.dump(H, open(cache, "w")); del net; torch.cuda.empty_cache()
    return H


def build_pairs(a, rng, H):
    out = []
    for l in open(a.train, encoding="utf-8"):
        e = json.loads(l); us = units(e)
        gold = [i for i, (pi, si, _) in enumerate(us) if e["passages"][pi]["gold"][si]]
        if not gold:
            continue
        gs = set(gold); hn = H.get(e["id"], []); neg = [i for i in range(len(us)) if i not in gs]
        for _ in range(a.draws):
            perm = gold[:]; rng.shuffle(perm)
            chain = perm[: rng.randint(0, len(perm))]          # uniform prefix length, incl. the complete set
            if hn and rng.random() < a.p_err:                   # a decoding error already in the chain
                chain.insert(rng.randint(0, len(chain)), rng.choice(hn[:4]))
            cs = set(chain)
            q = query_with_chain(e["question"], [us[i][2] for i in chain])
            pos = [i for i in gold if i not in cs]
            rest = [i for i in neg if i not in cs]
            cand = [i for i in hn if i not in cs][: a.hard] + rng.sample(rest, min(a.rand, len(rest)))
            out += [(q, us[i][2], 1.0) for i in pos] + [(q, us[i][2], 0.0) for i in dict.fromkeys(cand)]
    rng.shuffle(out)
    return out


def train(a):
    rng = random.Random(a.seed)
    H = hard_negatives(a)
    pairs = build_pairs(a, rng, H)
    print("pairs", len(pairs), "pos", sum(p[2] for p in pairs), flush=True)
    net = Net(a.backbone, a.max_len, train=True); m, tok = net.m, net.tok
    m.roberta.embeddings.word_embeddings.weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    steps = len(pairs) // a.bs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    m.gradient_checkpointing_enable(); m.train(); t0 = time.time()
    for i in range(steps):
        b = pairs[i * a.bs:(i + 1) * a.bs]
        enc = net.enc([x[0] for x in b], [x[1] for x in b])
        y = torch.tensor([x[2] for x in b], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = m(**enc).logits.view(-1)
        loss = F.binary_cross_entropy_with_logits(logit.float(), y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if i % 200 == 0:
            print(f"step {i}/{steps} loss {loss.item():.4f} {(time.time()-t0)/60:.1f}min", flush=True)
    m.save_pretrained(a.out); tok.save_pretrained(a.out)
    print("saved", flush=True)


# ------------------------------------------------------------------ decoding
class ChainDecoder:
    """ctx: 'chain' (own decoded chain), 'none' (C = empty at every step: independent scoring, same network),
    'random' (chain replaced by random units of equal length: tests that the *content* of the chain matters)."""

    def __init__(self, path, cand=24, max_steps=12, floor=0.05, ctx="chain", seed=0):
        self.net = Net(path); self.net.m.eval()
        self.cand, self.max_steps, self.floor, self.ctx = cand, max_steps, floor, ctx
        self.rng = random.Random(seed)

    def decode(self, ex):
        """returns step-0 probabilities and the decoded chain [(unit, p(unit | q, C), P(C sufficient | q))], where C is
        the chain before the step and sufficiency is the probability that no remaining unit is needed,
        P(S* within C | q, C) = prod_v (1 - p(v | q, C)), read off the same conditional network"""
        us = units(ex); txt = [u[2] for u in us]
        p0 = 1 / (1 + np.exp(-self.net.logits([ex["question"]] * len(us), txt)))
        pool = [int(j) for j in np.argsort(-p0)[: self.cand]]
        chain, p = [], p0[pool]
        while pool and len(chain) < self.max_steps:
            if chain and self.ctx != "none":
                ctx = [u for u, _, _ in chain]
                if self.ctx == "random":
                    rest = [j for j in range(len(us)) if j not in ctx]
                    ctx = self.rng.sample(rest, min(len(ctx), len(rest)))
                q = query_with_chain(ex["question"], [txt[c] for c in ctx])
                p = 1 / (1 + np.exp(-self.net.logits([q] * len(pool), [txt[j] for j in pool])))
            suff = float(np.exp(np.log1p(-np.clip(p, 0, 1 - 1e-6)).sum()))
            k = int(np.argmax(p))
            if chain and suff >= 0.95 and p[k] < self.floor:
                break
            chain.append((pool[k], float(p[k]), suff))
            pool.pop(k); p = np.delete(p, k)
        return p0.tolist(), chain


def sufficiency_stop(chain, tau=0.5):
    """Bayes decision on the event "the chain is already sufficient": decode while P(sufficient) < tau (tau = 1/2)"""
    n = next((i for i, (u, p, s) in enumerate(chain) if s >= tau), len(chain))
    return [c[0] for c in chain[: max(1, n)]]


def unit_stop(chain):
    """ablation: Bayes decision on the next unit alone (stop when the best remaining unit has p < 1/2)"""
    n = next((i for i, c in enumerate(chain) if c[1] < 0.5), len(chain))
    return [c[0] for c in chain[: max(1, n)]]


def price_stop(ex, chain, lam):
    us = units(ex); n = 0
    for i, (u, p, _) in enumerate(chain):
        if i > 0 and p - lam * len(us[u][2].split()) / 25.0 <= 0:
            break
        n = i + 1
    return [c[0] for c in chain[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ckpt/ftce")
    ap.add_argument("--train", default="data/proc/train.jsonl")
    ap.add_argument("--out", default="ckpt/ecd")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--hard", type=int, default=4)
    ap.add_argument("--rand", type=int, default=3)
    ap.add_argument("--p_err", type=float, default=0.3)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
