"""Sufficiency Potential Ascent (SPA): evidence compression as ascent on a learned potential over evidence sets.

One cross-encoder reads the question and a whole candidate evidence set S (rendered in document order) and returns a
scalar potential
    V(q, S) = f([q], [S])
that scores S as evidence for q. It is the only learned quantity. Compression is hill climbing on V over the
neighbourhood of the current set C,
    N(C) = { C + u : u not in C }  (add)   U   { C - v : v in C, |C| > 1 }  (remove)   U   { C }  (stay),
moving to the best neighbour and halting when C itself is the best one, i.e. at a local maximum of V. Each move strictly
increases V, so decoding never revisits a set and always terminates. Three decisions of earlier compressors become one:
what to keep (add), correcting an earlier choice (remove) and how much to keep (stay = C is already a local optimum).
There is no stop token, ratio, top-k, threshold or calibration split. With |S| = 1 the input is exactly that of an
independent sentence scorer, so V is initialised from one (FT-CE) and the step-0 ranking is that scorer.

Training: given a sufficient set S*, the correct moves are known in every state: add a missing gold unit, remove a
wrong unit, or stay iff C = S* (every correct move shortens the distance |C xor S*|). The loss is
    -log sum_{C' in N*(C)} softmax_{C'' in N(C)} V(q, C'')_{C'}
on teacher-forced states (prefixes of S*) and then, for R rounds, on the states the current climber visits itself
(DAgger), which is where remove moves are learned: they are labelled by the same rule, with no synthetic corruption.
"""
import argparse, json, os, random, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

SEP = " | "


def units(ex):
    return [(pi, si, f"{p['title']}: {s}") for pi, p in enumerate(ex["passages"]) for si, s in enumerate(p["sents"])]


def gold_of(ex, us=None):
    us = us or units(ex)
    return [i for i, (pi, si, _) in enumerate(us) if ex["passages"][pi]["gold"][si]]


def render(txt, S):
    """a set is rendered in document order, so V depends on the set and not on the path that built it"""
    return SEP.join(" ".join(txt[i].split()[:128]) for i in sorted(S))


class Potential:
    def __init__(self, path, max_len=512, train=False):
        self.tok = AutoTokenizer.from_pretrained(path)
        dt = torch.float32 if train else torch.bfloat16
        self.m = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=dt).cuda()
        self.max_len = max_len

    def enc(self, qs, ss):
        return self.tok(qs, ss, padding=True, truncation="only_second", max_length=self.max_len, return_tensors="pt").to("cuda")

    @torch.no_grad()
    def __call__(self, q, txt, sets, bs=64):
        """V(q, S) for every S in sets"""
        ss = [render(txt, S) for S in sets]; out = []
        for i in range(0, len(ss), bs):
            out += self.m(**self.enc([q] * len(ss[i:i + bs]), ss[i:i + bs])).logits.view(-1).float().tolist()
        return np.asarray(out)


# ------------------------------------------------------------------ moves
def neighbours(C, n, remove=True):
    """(kind, unit, set) for every move from C; 'stay' only for non-empty C, 'remove' only while |C| > 1"""
    cs = set(C); out = [("add", u, sorted(cs | {u})) for u in range(n) if u not in cs]
    if remove and len(C) > 1:
        out += [("remove", v, sorted(cs - {v})) for v in sorted(cs)]
    if C:
        out.append(("stay", None, sorted(cs)))
    return out


def correct(kind, u, C, gold):
    """a move is correct iff it shortens |C xor S*|; stay is correct iff C = S*"""
    if kind == "add":
        return u in gold
    if kind == "remove":
        return u not in gold
    return set(C) == gold


# ------------------------------------------------------------------ climbing
class Climber:
    def __init__(self, path, max_steps=16, remove=True, max_len=512, beam=1):
        self.V = Potential(path, max_len); self.V.m.eval()
        self.max_steps, self.remove, self.beam = max_steps, remove, beam

    def climb(self, ex):
        """greedy ascent from the empty set. Returns the singleton potentials z0 (step 0 = an independent ranking),
        the path [(kind, unit, V after the move)] and the potential of the returned set"""
        us = units(ex); txt = [u[2] for u in us]; q = ex["question"]; n = len(us)
        C, path, z0, v = [], [], None, None
        for _ in range(self.max_steps):
            nb = neighbours(C, n, self.remove)
            z = self.V(q, txt, [s for _, _, s in nb])
            if not C:
                z0 = z.tolist()
            k = int(np.argmax(z)); kind, u, S = nb[k]
            if kind == "stay":
                return z0, path, float(z[k]), True
            C, v = S, float(z[k]); path.append((kind, u, v))
        return z0, path, v, False

    def beam_climb(self, ex):
        """set-level beam search on V (possible because V scores whole sets): keep the B best distinct sets per step,
        return the best local maximum found"""
        us = units(ex); txt = [u[2] for u in us]; q = ex["question"]; n = len(us)
        beam = [((), None)]; best = (-np.inf, ()); seen = set()
        for _ in range(self.max_steps):
            cand = {}
            for C, vC in beam:
                nb = neighbours(list(C), n, self.remove)
                z = self.V(q, txt, [s for _, _, s in nb])
                k = int(np.argmax(z))
                if nb[k][0] == "stay" and z[k] > best[0]:
                    best = (float(z[k]), C)
                for (kind, u, S), zz in zip(nb, z):
                    S = tuple(S)
                    if kind != "stay" and S not in seen and (vC is None or zz > vC):
                        cand[S] = max(cand.get(S, -np.inf), float(zz))
            if not cand:
                break
            beam = sorted(cand.items(), key=lambda x: -x[1])[: self.beam]
            seen |= {S for S, _ in beam}
            if best[0] >= beam[0][1]:
                break
        return list(best[1]), best[0]


# ------------------------------------------------------------------ training
def teacher_states(e, rng, draws):
    gold = gold_of(e)
    out = []
    for _ in range(draws):
        perm = gold[:]; rng.shuffle(perm)
        out.append(perm[: rng.randint(0, len(perm))])
    return out


def group(e, C, hard, rng, n_cand, remove):
    """state C -> (question, candidate sets, targets): all correct moves, stay, every remove, then hard and random
    wrong adds up to n_cand candidates"""
    us = units(e); txt = [u[2] for u in us]; gold = set(gold_of(e, us)); cs = set(C)
    nb = neighbours(C, len(us), remove)
    pos = [m for m in nb if correct(m[0], m[1], C, gold)]
    rest = [m for m in nb if m not in pos]
    fixed = [m for m in rest if m[0] != "add"]
    adds = {m[1]: m for m in rest if m[0] == "add"}
    hn = [adds[j] for j in dict.fromkeys(hard) if j in adds]
    k = max(0, n_cand - len(pos) - len(fixed))
    hn = hn[: (k + 1) // 2]
    pool = [m for j, m in adds.items() if m not in hn]
    cand = pos + fixed + hn + rng.sample(pool, min(len(pool), k - len(hn)))
    y = [1.0 if m in pos else 0.0 for m in cand]
    return e["question"], [render(txt, m[2]) for m in cand], y


def marginal_hard(V, ex, cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    H = {}
    for n, e in enumerate(ex):
        us = units(e); txt = [u[2] for u in us]; g = set(gold_of(e, us))
        z = V(e["question"], txt, [[j] for j in range(len(us))], bs=128)
        H[e["id"]] = [int(j) for j in np.argsort(-z) if j not in g][:8]
        if n % 4000 == 0:
            print("hard", n, flush=True)
    json.dump(H, open(cache, "w"))
    return H


def fit(V, data, a, rng, tag):
    m = V.m
    m.roberta.embeddings.word_embeddings.weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    rng.shuffle(data)
    steps = len(data) // a.groups
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    m.gradient_checkpointing_enable(); m.train(); t0 = time.time(); run = []
    for i in range(steps):
        qs, ss, ys, seg = [], [], [], []
        for gi, (e, C, hard) in enumerate(data[i * a.groups:(i + 1) * a.groups]):
            q, s, y = group(e, C, hard, rng, a.cand, a.remove)
            qs += [q] * len(s); ss += s; ys += y; seg += [gi] * len(s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m(**V.enc(qs, ss)).logits.view(-1).float()
        y = torch.tensor(ys, device="cuda"); seg = torch.tensor(seg, device="cuda")
        G = int(seg.max()) + 1; loss = 0.0
        for gi in range(G):
            zg, yg = z[seg == gi], y[seg == gi]
            loss = loss + torch.logsumexp(zg, 0) - torch.logsumexp(zg.masked_fill(yg == 0, -1e4), 0)
        loss = loss / G
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); run.append(loss.item())
        if i % 200 == 0:
            print(f"[{tag}] step {i}/{steps} loss {np.mean(run[-200:]):.4f} {(time.time()-t0)/60:.1f}min", flush=True)
    m.eval()


@torch.no_grad()
def rollout(V, e, max_steps, remove, n_hard=8):
    """every state the current climber visits, with its own top-scored wrong adds as hard negatives"""
    us = units(e); txt = [u[2] for u in us]; gold = set(gold_of(e, us)); n = len(us)
    C, out = [], []
    for _ in range(max_steps):
        nb = neighbours(C, n, remove)
        z = V(e["question"], txt, [s for _, _, s in nb])
        order = [nb[k] for k in np.argsort(-z)]
        out.append((e, list(C), [m[1] for m in order if m[0] == "add" and m[1] not in gold][:n_hard]))
        kind, u, S = order[0]
        if kind == "stay":
            break
        C = S
    return out


def train(a):
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    ex = [json.loads(l) for l in open(a.train, encoding="utf-8")]
    ex = [e for e in ex if gold_of(e)][: a.limit or None]
    print("train questions", len(ex), "remove moves", a.remove, flush=True)
    V = Potential(a.backbone, a.max_len, train=True); V.m.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        H = marginal_hard(V, ex, a.hard_cache)
    data = [(e, C, [j for j in H.get(e["id"], []) if j not in C]) for e in ex for C in teacher_states(e, rng, a.draws)]
    print("round0 states", len(data), flush=True)
    fit(V, data, a, rng, "r0")
    V.m.save_pretrained(a.out + "_r0"); V.tok.save_pretrained(a.out + "_r0")
    for r in range(1, a.rounds):
        sub = rng.sample(ex, min(len(ex), a.dagger_n)); roll = []; t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for n, e in enumerate(sub):
                roll += rollout(V, e, a.max_steps, a.remove)
                if n % 1000 == 0:
                    print(f"[r{r}] rollout {n}/{len(sub)} states {len(roll)} {(time.time()-t0)/60:.1f}min", flush=True)
        off = sum(1 for e, C, _ in roll if not set(C) <= set(gold_of(e)))
        print(f"[r{r}] rollout states {len(roll)} off-oracle {off}", flush=True)
        data = roll + [(e, C, [j for j in H.get(e["id"], []) if j not in C]) for e in ex for C in teacher_states(e, rng, 1)]
        fit(V, data, a, rng, f"r{r}")
        tag = a.out if r == a.rounds - 1 else f"{a.out}_r{r}"
        V.m.save_pretrained(tag); V.tok.save_pretrained(tag)
    print("saved", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ckpt/ftce")
    ap.add_argument("--train", default="data/proc/train.jsonl")
    ap.add_argument("--out", default="ckpt/spa")
    ap.add_argument("--hard_cache", default="ckpt/ecd_hardneg.json")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--cand", type=int, default=12)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--dagger_n", type=int, default=10000)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--remove", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); a.remove = bool(a.remove)
    train(a)


if __name__ == "__main__":
    main()
