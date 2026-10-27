"""Evidence Set Decoding (ESD): evidence compression as imitation of the oracle set-decoding policy.

A compressor builds its output one unit at a time. At state (q, C) -- the question and the set of units kept so far --
it either appends a unit or halts. The oracle policy is known exactly: append any unit of the sufficient set S* that
is not yet in C, and halt (action END) iff S* is contained in C. ESD learns this policy with one cross-encoder
    pi(a | q, C)  proportional to  exp f([q <ctx> C], a),     a in (U minus C)  plus  {END}    (END only when C is non-empty)
where C is rendered as a set (document order) and END is one more candidate string read by the same network.
Training maximises the likelihood of the oracle's action set, log sum_{a in A*(C)} pi(a | q, C), on
  round 0: teacher-forced states (random orderings and prefix lengths of S*),
  round 1: the states the round-0 policy visits itself (DAgger), with the oracle's actions as labels and the
           policy's own top-scored wrong units as negatives.
Decoding is greedy: a_t = argmax pi(. | q, C_t); the compression ends when END wins. The length of the compression is a
decision of the same policy that chooses the units -- there is no ratio, top-k, threshold or calibration split.
"""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

CTX_SEP = " <ctx> "
END = "[END] no further evidence is needed"


def units(ex):
    return [(pi, si, f"{p['title']}: {s}") for pi, p in enumerate(ex["passages"]) for si, s in enumerate(p["sents"])]


def gold_of(ex, us=None):
    us = us or units(ex)
    return [i for i, (pi, si, _) in enumerate(us) if ex["passages"][pi]["gold"][si]]


def state(q, txt, C):
    """the state is a set: chain units are rendered in document order, not selection order"""
    return q if not C else q + CTX_SEP + " | ".join(txt[i] for i in sorted(C))


class Net:
    def __init__(self, path, max_len=512, train=False):
        self.tok = AutoTokenizer.from_pretrained(path)
        dt = torch.float32 if train else torch.bfloat16
        self.m = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=dt).cuda()
        self.max_len = max_len

    def enc(self, qs, us):
        # only the state may be truncated; a candidate is clipped to 128 words (a few units are longer than the input)
        us = [" ".join(u.split()[:128]) for u in us]
        return self.tok(qs, us, padding=True, truncation="only_first", max_length=self.max_len, return_tensors="pt").to("cuda")

    @torch.no_grad()
    def logits(self, qs, us, bs=96):
        out = []
        for i in range(0, len(qs), bs):
            out += self.m(**self.enc(qs[i:i + bs], us[i:i + bs])).logits.view(-1).float().tolist()
        return np.asarray(out)


# ------------------------------------------------------------------ states
def teacher_states(e, rng, draws, H):
    """round 0: C = a random prefix of a random ordering of S*; negatives are the marginally most confusable units"""
    us = units(e); gold = gold_of(e, us)
    if not gold:
        return []
    out = []
    for _ in range(draws):
        perm = gold[:]; rng.shuffle(perm)
        C = perm[: rng.randint(0, len(perm))]
        out.append({"q": e["question"], "C": C, "hard": [j for j in H.get(e["id"], []) if j not in C]})
    return out


def oracle_actions(gold, C):
    rest = [g for g in gold if g not in set(C)]
    return rest if rest else ["END"]


def group(e, st, rng, n_cand):
    """one training group: the state, the oracle action set (targets), and n_cand candidate actions in total"""
    us = units(e); txt = [u[2] for u in us]; gold = gold_of(e, us); C = st["C"]; cs = set(C)
    tgt = oracle_actions(gold, C)
    cand = list(tgt) + ([] if "END" in tgt or not C else ["END"])
    neg = [j for j in dict.fromkeys(st["hard"]) if j not in cs and j not in gold]
    rest = [j for j in range(len(us)) if j not in cs and j not in gold and j not in neg]
    k = max(0, n_cand - len(cand))
    hn = neg[: (k + 1) // 2]
    cand += hn + rng.sample(rest, min(len(rest), k - len(hn)))
    y = [1.0 if (c in tgt) else 0.0 for c in cand]
    return state(e["question"], txt, C), [END if c == "END" else txt[c] for c in cand], y


def marginal_hard(net, path, out_cache):
    """step-0 confusable units of every training question under the initial network"""
    if os.path.exists(out_cache):
        return {k: v for k, v in json.load(open(out_cache)).items()}
    H = {}
    for n, l in enumerate(open(path, encoding="utf-8")):
        e = json.loads(l); us = units(e); g = set(gold_of(e, us))
        z = net.logits([e["question"]] * len(us), [u[2] for u in us], bs=128)
        H[e["id"]] = [int(j) for j in np.argsort(-z) if j not in g][:8]
        if n % 4000 == 0:
            print("hard", n, flush=True)
    json.dump(H, open(out_cache, "w"))
    return H


# ------------------------------------------------------------------ training
def fit(net, data, a, rng, tag):
    """data: list of (example, state). Loss: -log sum_{a in A*} softmax(f)_a over each group's candidates."""
    m = net.m
    m.roberta.embeddings.word_embeddings.weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    rng.shuffle(data)
    steps = len(data) // a.groups
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    m.gradient_checkpointing_enable(); m.train(); t0 = time.time(); run = []
    for i in range(steps):
        qs, cs, ys, seg = [], [], [], []
        for gi, (e, st) in enumerate(data[i * a.groups:(i + 1) * a.groups]):
            q, c, y = group(e, st, rng, a.cand)
            qs += [q] * len(c); cs += c; ys += y; seg += [gi] * len(c)
        enc = net.enc(qs, cs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m(**enc).logits.view(-1).float()
        y = torch.tensor(ys, device="cuda"); seg = torch.tensor(seg, device="cuda")
        loss = 0.0
        for gi in range(int(seg.max()) + 1):
            zg, yg = z[seg == gi], y[seg == gi]
            loss = loss + torch.logsumexp(zg, 0) - torch.logsumexp(zg.masked_fill(yg == 0, -1e4), 0)
        loss = loss / (int(seg.max()) + 1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); run.append(loss.item())
        if i % 200 == 0:
            print(f"[{tag}] step {i}/{steps} loss {np.mean(run[-200:]):.4f} {(time.time()-t0)/60:.1f}min", flush=True)
    m.eval()


def rollout_states(net, e, max_steps, n_hard=8):
    """round 1 (DAgger): every state the current policy visits on its own greedy path, labelled by the oracle;
    the policy's own highest-scored wrong units at that state become its negatives"""
    us = units(e); txt = [u[2] for u in us]; gold = set(gold_of(e, us))
    C, out = [], []
    for _ in range(max_steps):
        pool = [j for j in range(len(us)) if j not in C]
        cand = pool + (["END"] if C else [])
        z = net.logits([state(e["question"], txt, C)] * len(cand), [END if c == "END" else txt[c] for c in cand])
        order = [cand[k] for k in np.argsort(-z)]
        out.append({"q": e["question"], "C": list(C), "hard": [c for c in order if c != "END" and c not in gold][:n_hard]})
        a = order[0]
        if a == "END" or not pool:
            break
        C.append(a)
    return out


def train(a):
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    ex = [json.loads(l) for l in open(a.train, encoding="utf-8")]
    ex = [e for e in ex if gold_of(e)][: a.limit or None]
    print("train questions", len(ex), flush=True)
    net = Net(a.backbone, a.max_len, train=True); net.m.eval()
    H = marginal_hard(net, a.train, a.hard_cache)
    # round 0: teacher forcing over orderings of S*
    data = [(e, st) for e in ex for st in teacher_states(e, rng, a.draws, H)]
    print("round0 states", len(data), flush=True)
    fit(net, data, a, rng, "r0")
    net.m.save_pretrained(a.out + "_r0"); net.tok.save_pretrained(a.out + "_r0")
    if a.rounds < 2:
        return
    # round 1: DAgger on the policy's own states (half of the questions), mixed with fresh teacher states
    sub = rng.sample(ex, min(len(ex), a.dagger_n)); roll = []; t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for n, e in enumerate(sub):
            roll += [(e, st) for st in rollout_states(net, e, a.max_steps)]
            if n % 1000 == 0:
                print(f"rollout {n}/{len(sub)} states {len(roll)} {(time.time()-t0)/60:.1f}min", flush=True)
    off = sum(1 for e, st in roll if not set(st["C"]) <= set(gold_of(e)))
    print("round1 rollout states", len(roll), "off-oracle states", off, flush=True)
    data = roll + [(e, st) for e in ex for st in teacher_states(e, rng, 1, H)]
    fit(net, data, a, rng, "r1")
    net.m.save_pretrained(a.out); net.tok.save_pretrained(a.out)
    print("saved", flush=True)


# ------------------------------------------------------------------ decoding
class SetDecoder:
    """ctx: 'set'    -- the policy reads its own decoded set (ESD),
            'random' -- the set is replaced by random units of equal size at every step (tests that the content of
                        the set, not the longer input, is what the policy uses).
    The marginal control (C = empty at every step) needs no decoding: it is the ranking of the empty-state logits z0."""

    def __init__(self, path, max_steps=12, ctx="set", seed=0, max_len=512):
        self.net = Net(path, max_len); self.net.m.eval()
        self.max_steps, self.ctx = max_steps, ctx
        self.rng = random.Random(seed)

    def decode(self, ex):
        """returns the empty-state logits z0 of all units, the decoded chain [(unit, pi(unit), pi(END))] and pi(END)
        at the state where END won (None if the step cap was reached). Each state scores all remaining units and END
        in one batch."""
        us = units(ex); txt = [u[2] for u in us]; q = ex["question"]
        C, chain, z0 = [], [], None
        while len(C) < min(self.max_steps, len(us)):
            pool = [j for j in range(len(us)) if j not in C]
            ctx = C
            if self.ctx == "random" and C:
                ctx = self.rng.sample(pool, min(len(C), len(pool)))
            z = self.net.logits([state(q, txt, ctx)] * (len(pool) + 1), [txt[j] for j in pool] + [END])
            if not C:
                z0 = z[:-1].tolist()
                z[-1] = -np.inf  # END is not an action in the empty state (S* is non-empty)
            p = np.exp(z - z.max()); p /= p.sum()
            k = int(np.argmax(z))
            if k == len(pool):
                return z0, chain, float(p[-1])
            chain.append((pool[k], float(p[k]), float(p[-1])))
            C.append(pool[k])
        return z0, chain, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ckpt/ftce")
    ap.add_argument("--train", default="data/proc/train.jsonl")
    ap.add_argument("--out", default="ckpt/esd")
    ap.add_argument("--hard_cache", default="ckpt/ecd_hardneg.json")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--cand", type=int, default=12)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--dagger_n", type=int, default=12000)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
