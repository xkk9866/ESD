"""Answer metrics (EM/F1/Acc) and evidence-fidelity metrics (SF-Rec, Ans-Rec, QP, Prov)."""
import re, string
from collections import Counter
from units import qualifier_cues


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, answers):
    return float(any(norm(pred) == norm(a) for a in answers))


def _f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if p in (["yes"], ["no"]) or g in (["yes"], ["no"]):
        return float(p == g)
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def f1(pred, answers):
    return max(_f1(pred, a) for a in answers)


def acc(pred, answers):
    np_ = " " + norm(pred) + " "
    return float(any(norm(a) and (" " + norm(a) + " ") in np_ for a in answers))


_W = re.compile(r"\w+(?:'t)?")


def words(t):
    return [w.lower() for w in _W.findall(t)]


def sent_retained(sent, ctx_words_set, thr=0.8):
    w = words(sent)
    if not w:
        return False
    return sum(x in ctx_words_set for x in w) / len(w) >= thr


def ngram_set(ws, n):
    return set(tuple(ws[i:i + n]) for i in range(len(ws) - n + 1))


def fidelity(ex, ctx, kept=None):
    """kept: list of [pid, sid] for extractive methods (exact), else None (alignment by word recall)."""
    cw = words(ctx); cset = set(cw)
    gold = [(pi, si, s) for pi, p in enumerate(ex["passages"]) for si, (s, g) in enumerate(zip(p["sents"], p["gold"])) if g]
    if kept is not None:
        ks = set(map(tuple, kept))
        ret = [(pi, si) in ks for pi, si, _ in gold]
    else:
        ret = [sent_retained(s, cset) for _, _, s in gold]
    sf = sum(ret) / max(1, len(gold))
    # qualifier preservation: among retained gold sentences that carry cues, all cues present in compressed evidence
    qn, qk = 0, 0
    for (pi, si, s), r in zip(gold, ret):
        cues = qualifier_cues(s)
        if not cues:
            continue
        qn += 1
        if kept is not None:
            qk += int(r)
        else:
            # strict check for non-extractive output: content retained AND every cue survives
            qk += int(r and all(c.lower() in cset or c in ctx for c in cues))
    qp = qk / qn if qn else None
    # strict qualifier preservation over ALL gold cue-bearing sentences (dropping the sentence also loses the cue)
    # provenance: fraction of output words covered by source 4-grams
    src = words(" ".join(" ".join([p["title"]] + p["sents"]) for p in ex["passages"]))
    s4 = ngram_set(src, 4)
    cov = [False] * len(cw)
    for i in range(len(cw) - 3):
        if tuple(cw[i:i + 4]) in s4:
            for j in range(i, i + 4):
                cov[j] = True
    prov = sum(cov) / len(cw) if cw else 1.0
    nctx = " " + " ".join(cw) + " "
    ansrec = float(any(norm(a) and (" " + norm(a) + " ") in " " + norm(ctx) + " " for a in ex["answers"]))
    return {"sf_rec": sf, "qp": qp, "qp_n": qn, "prov": prov, "ans_rec": ansrec}
