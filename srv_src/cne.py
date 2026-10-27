"""Conditional Necessity Estimation (CNE): an evidence dependency network for RAG compression.

One cross-encoder learns the whole family of conditionals  p(u in S* | q, C)  for any kept context C:
  input = [q  <ctx> title: s | title: s ...]  [SEP]  [title: u]
Training is pseudo-likelihood over randomly sampled contexts (C = empty recovers the independent scorer);
inference is a fixed-point (mean-field / ICM) iteration: score -> select context -> re-score conditioned on it.
"""
import argparse, json, random, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

CTX_SEP = " <ctx> "


def units(ex):
    return [(pi, si, f"{p['title']}: {s}") for pi, p in enumerate(ex["passages"]) for si, s in enumerate(p["sents"])]


def query_with_ctx(q, ctx_texts):
    return q if not ctx_texts else q + CTX_SEP + " | ".join(ctx_texts)


# ------------------------------------------------------------------ training
def sample_context(rng, gold, neg, a):
    """random kept context: each gold unit w.p. 1/2, plus a few distractors; empty w.p. p_empty"""
    if rng.random() < a.p_empty:
        return []
    c = [g for g in gold if rng.random() < 0.5] + rng.sample(neg, min(len(neg), rng.randint(0, a.max_ctx_neg)))
    rng.shuffle(c)
    return c[: a.max_ctx]


def build_pairs(a, rng):
    out = []
    for l in open(a.train, encoding="utf-8"):
        e = json.loads(l)
        us = units(e)
        gold = [i for i, (pi, si, _) in enumerate(us) if e["passages"][pi]["gold"][si]]
        neg = [i for i in range(len(us)) if i not in set(gold)]
        for _ in range(a.draws):
            c = sample_context(rng, gold, neg, a)
            cs = set(c)
            q = query_with_ctx(e["question"], [us[i][2] for i in c])
            pos = [i for i in gold if i not in cs]
            ng = [i for i in neg if i not in cs]
            rng.shuffle(ng)
            out += [(q, us[i][2], 1.0) for i in pos] + [(q, us[i][2], 0.0) for i in ng[: a.neg]]
    rng.shuffle(out)
    return out


def train(a):
    rng = random.Random(a.seed)
    pairs = build_pairs(a, rng)
    print("pairs", len(pairs), flush=True)
    tok = AutoTokenizer.from_pretrained(a.backbone)
    m = AutoModelForSequenceClassification.from_pretrained(a.backbone).cuda()
    m.roberta.embeddings.word_embeddings.weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    steps = len(pairs) // a.bs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    m.gradient_checkpointing_enable(); m.train(); t0 = time.time()
    for i in range(steps):
        b = pairs[i * a.bs:(i + 1) * a.bs]
        enc = tok([q for q, _, _ in b], [s for _, s, _ in b], padding=True, truncation="longest_first",
                  max_length=a.max_len, return_tensors="pt").to("cuda")
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


# ------------------------------------------------------------------ inference
class CNEScorer:
    """Fixed-point conditional scoring. iters=0 -> independent scorer (empty context).
    ctx_mode: 'self' (the model's own current selection), 'random' (ablation), 'oracle' (gold units, upper bound)."""

    def __init__(self, path, iters=2, top=48, ctx_k=4, ctx_thr=0.0, ctx_mode="self", max_len=384, seed=0):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
        self.iters, self.top, self.ctx_k, self.ctx_thr, self.ctx_mode, self.max_len = iters, top, ctx_k, ctx_thr, ctx_mode, max_len
        self.rng = random.Random(seed)
        self.cache = {}

    @torch.no_grad()
    def _score(self, qs, cands):
        out = []
        for i in range(0, len(qs), 64):
            b = self.tok(qs[i:i + 64], cands[i:i + 64], padding=True, truncation="longest_first",
                         max_length=self.max_len, return_tensors="pt").to("cuda")
            out += self.m(**b).logits.view(-1).float().tolist()
        return out

    def _context(self, ex, us, z):
        if self.ctx_mode == "oracle":
            return [i for i, (pi, si, _) in enumerate(us) if ex["passages"][pi]["gold"][si]]
        if self.ctx_mode == "random":
            return self.rng.sample(range(len(us)), min(self.ctx_k, len(us)))
        order = np.argsort(-z)
        return [int(j) for j in order[: self.ctx_k] if z[j] > self.ctx_thr] or [int(order[0])]

    def __call__(self, ex):
        if ex["id"] not in self.cache:
            self.cache[ex["id"]] = self.infer(ex)
        return self.cache[ex["id"]]

    def infer(self, ex):
        us = units(ex)
        txt = [u[2] for u in us]
        z = np.asarray(self._score([ex["question"]] * len(us), txt))
        for _ in range(self.iters):
            ctx = self._context(ex, us, z)
            cand = [int(j) for j in np.argsort(-z)[: self.top]]
            # pseudo-likelihood conditional: each unit is scored given the context without itself
            qs = [query_with_ctx(ex["question"], [txt[c] for c in ctx if c != j]) for j in cand]
            zc = self._score(qs, [txt[j] for j in cand])
            z = z.copy()
            z[cand] = zc
        return z.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ckpt/ftce")
    ap.add_argument("--train", default="data/proc/train.jsonl")
    ap.add_argument("--out", default="ckpt/cne")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--neg", type=int, default=6)
    ap.add_argument("--p_empty", type=float, default=0.25)
    ap.add_argument("--max_ctx", type=int, default=5)
    ap.add_argument("--max_ctx_neg", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
