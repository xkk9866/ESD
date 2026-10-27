"""Baseline evidence compressors. Each writes results/compressed/<ds>/<method>.jsonl with {id, context, kept?, time}."""
import argparse, json, os, re, time
import numpy as np, torch

M = "models"
DATASETS = ["nq", "hotpotqa", "musique", "2wiki"]


def load(ds, n):
    return [json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")][:n]


def sent_list(ex):
    return [(pi, si, s) for pi, p in enumerate(ex["passages"]) for si, s in enumerate(p["sents"])]


def render_full(ex):
    return "\n".join(f"{p['title']}: {' '.join(p['sents'])}" for p in ex["passages"])


def nwords_full(ex):
    return sum(len((p["title"] + ":").split()) + sum(len(s.split()) for s in p["sents"]) for p in ex["passages"])


def render_kept(ex, kept):
    by = {}
    for pi, si in sorted(kept):
        by.setdefault(pi, []).append(ex["passages"][pi]["sents"][si])
    return "\n".join(f"{ex['passages'][pi]['title']}: {' '.join(v)}" for pi, v in sorted(by.items()))


def greedy_budget(ex, scores, budget):
    """Pick sentences by descending score until the word budget (title words counted once per passage)."""
    sl = sent_list(ex)
    order = np.argsort(-np.asarray(scores))
    used, kept, titled = 0, [], set()
    for j in order:
        pi, si, s = sl[j]
        c = len(s.split()) + (0 if pi in titled else len((ex["passages"][pi]["title"] + ":").split()))
        if used + c > budget:
            continue
        used += c; kept.append((pi, si)); titled.add(pi)
    if not kept:  # always keep at least the top sentence
        pi, si, _ = sl[order[0]]; kept = [(pi, si)]
    return kept


def budgets_for(ds, exs, mode):
    if mode == "match" or mode.startswith("m_"):  # per-query word budget of a reference compressor (m_<name>)
        refn = "vericomp" if mode == "match" else mode[2:]
        ref = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/compressed/{ds}/{refn}.jsonl", encoding="utf-8")}
        return [len(ref[e["id"]]["context"].split()) for e in exs]
    r = int(mode[1:])
    return [nwords_full(e) / r for e in exs]


# ---------------- scorers ----------------
class BM25Scorer:
    def __call__(self, ex):
        from rank_bm25 import BM25Okapi
        tok = lambda t: re.findall(r"\w+", t.lower())
        sl = sent_list(ex)
        bm = BM25Okapi([tok(ex["passages"][pi]["title"] + " " + s) for pi, si, s in sl])
        return bm.get_scores(tok(ex["question"])).tolist()


class CrossEncScorer:
    def __init__(self, path):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def __call__(self, ex):
        sl = sent_list(ex)
        pairs = [[ex["question"], f"{ex['passages'][pi]['title']}: {s}"] for pi, si, s in sl]
        out = []
        for i in range(0, len(pairs), 128):
            b = self.tok(pairs[i:i + 128], padding=True, truncation=True, max_length=192, return_tensors="pt").to("cuda")
            out += self.m(**b).logits.view(-1).float().tolist()
        return out


class RecompExtScorer:
    """RECOMP extractive compressor (Contriever-style dual encoder, mean pooling, dot product)."""
    def __init__(self, path):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModel.from_pretrained(path).cuda().eval().half()

    def enc(self, texts):
        b = self.tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        h = self.m(**b).last_hidden_state
        msk = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return (h * msk).sum(1) / msk.sum(1)

    @torch.no_grad()
    def __call__(self, ex):
        sl = sent_list(ex)
        q = self.enc([ex["question"]])
        out = []
        for i in range(0, len(sl), 128):
            s = self.enc([f"{ex['passages'][pi]['title']} {t}" for pi, si, t in sl[i:i + 128]])
            out += (s @ q.T).view(-1).float().tolist()
        return out


def price_select(ex, probs, lam):
    """independent unit selection at token price lam: keep s iff p_s > lam * len_s / 25 (title cost counted in output)"""
    sl = sent_list(ex)
    kept = [(pi, si) for (pi, si, t), q in zip(sl, probs) if q - lam * len(t.split()) / 25.0 > 0]
    if not kept:
        j = int(np.argmax(probs)); kept = [(sl[j][0], sl[j][1])]
    return kept


def kept_words(ex, kept):
    return len(render_kept(ex, kept).split())


def run_price(name, scorer, ds, exs, ratios):
    """corpus-level price mode: one token price per dataset, chosen on the calibration split to meet the average budget"""
    LAMS = [0.0] + list(np.logspace(-4, np.log10(5.0), 60))
    sig = lambda v: (1 / (1 + np.exp(-np.asarray(v, dtype=np.float64)))).tolist()
    cal = [json.loads(l) for l in open(f"data/proc/calib_{ds}.jsonl", encoding="utf-8")]
    cp = [sig(scorer(e)) for e in cal]
    mc = np.array([[kept_words(e, price_select(e, q, lam)) for lam in LAMS] for e, q in zip(cal, cp)]).mean(0)
    full = np.mean([nwords_full(e) for e in cal])
    for r in ratios:
        ok = [lam for lam, c in zip(LAMS, mc) if c <= full / r]
        lam = min(ok) if ok else LAMS[-1]
        print(name, ds, "p%d" % r, "lambda", lam, flush=True)
        with open(f"results/compressed/{ds}/{name}_p{r}.jsonl", "w", encoding="utf-8") as f:
            for e in exs:
                t0 = time.time(); q = sig(scorer(e))
                kept = price_select(e, q, lam)
                f.write(json.dumps({"id": e["id"], "context": render_kept(e, kept), "kept": kept, "time": time.time() - t0}) + "\n")


def run_extractive(name, scorer, ds, exs, modes):
    pr = [int(m[1:]) for m in modes if m.startswith("p")]
    if pr:
        run_price(name, scorer, ds, exs, pr)
    modes = [m for m in modes if not m.startswith("p")]
    if "ok" in modes:  # length oracle: the top-|S*| units (uses the gold count of each test question)
        modes.remove("ok")
        with open(f"results/compressed/{ds}/{name}_ok.jsonl", "w", encoding="utf-8") as f:
            for e in exs:
                t0 = time.time(); sc = np.asarray(scorer(e)); sl = sent_list(e)
                n = max(1, sum(sum(p["gold"]) for p in e["passages"]))
                kept = [(sl[j][0], sl[j][1]) for j in np.argsort(-sc)[:n]]
                f.write(json.dumps({"id": e["id"], "context": render_kept(e, kept), "kept": kept, "time": time.time() - t0}) + "\n")
    for mode in modes:
        bud = budgets_for(ds, exs, mode)
        with open(f"results/compressed/{ds}/{name}_{mode}.jsonl", "w", encoding="utf-8") as f:
            for e, b in zip(exs, bud):
                t0 = time.time(); sc = scorer(e); torch.cuda.synchronize() if torch.cuda.is_available() else None
                kept = greedy_budget(e, sc, b)
                f.write(json.dumps({"id": e["id"], "context": render_kept(e, kept), "kept": kept, "time": time.time() - t0}) + "\n")


def run_trunc(ds, exs, modes):
    for mode in modes:
        bud = budgets_for(ds, exs, mode)
        with open(f"results/compressed/{ds}/trunc_{mode}.jsonl", "w", encoding="utf-8") as f:
            for e, b in zip(exs, bud):
                w = render_full(e).split(" ")
                f.write(json.dumps({"id": e["id"], "context": " ".join(w[: int(b)]), "time": 0.0}) + "\n")


def simple(ds, exs, name, fn):
    with open(f"results/compressed/{ds}/{name}.jsonl", "w", encoding="utf-8") as f:
        for e in exs:
            t0 = time.time(); c = fn(e)
            f.write(json.dumps({"id": e["id"], "context": c, "time": time.time() - t0}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--modes", default="match,r2,r4,r8")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--points", default="", help="operating points: rates (LLMLingua) or thresholds (Provence)")
    a = ap.parse_args()
    modes = a.modes.split(",")
    a.points = [float(x) for x in a.points.split(",")] if a.points else ([0.33] if a.method == "llmlingua2" else [0.25] if a.method == "longllmlingua" else [0.1])
    dss = a.datasets.split(",")
    m = a.method
    if m in ("bge", "ftce"):
        scorer = CrossEncScorer(a.ckpt or f"{M}/bge-reranker-v2-m3")
    elif m == "bm25":
        scorer = BM25Scorer()
    elif m == "llmlingua2":
        from llmlingua import PromptCompressor
        pc = PromptCompressor(model_name=f"{M}/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True, device_map="cuda")
    elif m == "longllmlingua":
        from llmlingua import PromptCompressor
        from transformers import BitsAndBytesConfig
        pc = PromptCompressor(model_name=a.ckpt, device_map="cuda",
                              model_config={"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)})
    elif m == "provence":
        from transformers import AutoModel
        prov = AutoModel.from_pretrained(f"{M}/provence-reranker-debertav3-v1", trust_remote_code=True).cuda().eval()
    for ds in dss:
        os.makedirs(f"results/compressed/{ds}", exist_ok=True)
        exs = load(ds, a.n)
        if m == "full":
            simple(ds, exs, "full", render_full)
        elif m == "closedbook":
            simple(ds, exs, "closedbook", lambda e: "")
        elif m == "trunc":
            run_trunc(ds, exs, modes)
        elif m == "recomp_ext":
            path = a.ckpt or ("fangyuan/nq_extractive_compressor" if ds == "nq" else "fangyuan/hotpotqa_extractive_compressor")
            run_extractive(m, RecompExtScorer(path), ds, exs, modes)
        elif m in ("bm25", "bge", "ftce"):
            run_extractive(a.name or m, scorer, ds, exs, modes)
        elif m == "llmlingua2":
            for rate in a.points:
                def f(e, rate=rate):
                    docs = [f"{p['title']}: {' '.join(p['sents'])}" for p in e["passages"]]
                    return pc.compress_prompt(docs, rate=rate, force_tokens=["\n", "?", ".", "!"], drop_consecutive=True)["compressed_prompt"]
                simple(ds, exs, "llmlingua2" if rate == 0.33 else f"llmlingua2_q{rate}", f)
        elif m == "longllmlingua":
            for rate in a.points:
                def f(e, rate=rate):
                    docs = [f"{p['title']}: {' '.join(p['sents'])}" for p in e["passages"]]
                    return pc.compress_prompt(docs, question=e["question"], rate=rate, condition_in_question="after_condition",
                                              reorder_context="sort", dynamic_context_compression_ratio=0.3, condition_compare=True,
                                              context_budget="+100", rank_method="longllmlingua")["compressed_prompt"]
                simple(ds, exs, "longllmlingua" if rate == 0.25 else f"longllmlingua_q{rate}", f)
        elif m == "provence":
            for th in a.points:
                def f(e, th=th):
                    docs = [f"{p['title']}: {' '.join(p['sents'])}" for p in e["passages"]]
                    out = prov.process(e["question"], [docs], threshold=th, always_select_title=True, enable_warnings=False)
                    return "\n".join(x for x in out["pruned_context"][0] if x.strip())
                simple(ds, exs, "provence" if th == 0.1 else f"provence_q{th}", f)


if __name__ == "__main__":
    main()
