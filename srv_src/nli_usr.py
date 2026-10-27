"""Unsupported-Strengthening Rate (USR): fraction of compressed-evidence sentences that are either
(a) not entailed by their best-aligned source window (NLI), or (b) drop a negation/hedge/condition cue present in that source."""
import argparse, glob, json, os, re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from units import split_sentences, qualifier_cues, NEG, HEDGE, COND
from metrics import words

SEM = NEG | HEDGE | COND | {"n't"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", default="models/nli-deberta-v3-base")
    ap.add_argument("--datasets", default="nq,hotpotqa,musique,2wiki")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--methods", default="", help="comma-separated subset (default: every compressed file)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.nli)
    m = AutoModelForSequenceClassification.from_pretrained(a.nli).cuda().eval().half()
    lab = {v.lower(): k for k, v in m.config.id2label.items()}
    ent_id = lab.get("entailment")
    res = json.load(open("results/usr.json")) if os.path.exists("results/usr.json") else {}
    for ds in a.datasets.split(","):
        test = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")}
        for fp in sorted(glob.glob(f"results/compressed/{ds}/*.jsonl")):
            meth = os.path.basename(fp)[:-6]
            if meth in ("closedbook",) or (a.methods and meth not in a.methods.split(",")):
                continue
            recs = [json.loads(l) for l in open(fp, encoding="utf-8")][: a.n]
            bad, tot = 0, 0
            for r in recs:
                e = test[r["id"]]
                src = [s for p in e["passages"] for s in [p["title"]] + p["sents"]]
                srcw = [set(words(s)) for s in src]
                srcjoin = " ".join(src)
                outs = [s for line in r["context"].split("\n") for s in split_sentences(line)]
                hyps, prem, flags = [], [], []
                for s in outs:
                    body = s.split(": ", 1)[1] if ": " in s[:120] and s.split(": ", 1)[0] in srcjoin else s
                    if len(body.split()) < 3:
                        continue
                    tot += 1
                    if body in srcjoin:
                        continue  # verbatim span of the source: supported, nothing strengthened
                    w = set(words(body))
                    j = max(range(len(src)), key=lambda k: len(w & srcw[k]) / (len(w) + 1e-9))
                    window = " ".join(src[max(0, j - 1): j + 2])
                    lost = {c for c in qualifier_cues(src[j]) if c in SEM} - set(words(body)) - {c for c in qualifier_cues(body)}
                    overlap = len(w & srcw[j]) / max(1, len(w))
                    if lost and overlap > 0.5:
                        bad += 1; continue
                    hyps.append(body); prem.append(window)
                for i in range(0, len(hyps), 32):
                    b = tok(prem[i:i + 32], hyps[i:i + 32], truncation=True, max_length=384, padding=True, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        pr = torch.softmax(m(**b).logits.float(), -1)[:, ent_id]
                    bad += int((pr < 0.5).sum().item())
            res.setdefault(ds, {})[meth] = 100 * bad / max(1, tot)
            print(ds, meth, f"USR {res[ds][meth]:.1f} over {tot} sents", flush=True)
            json.dump(res, open("results/usr.json", "w"), indent=1)


if __name__ == "__main__":
    main()
