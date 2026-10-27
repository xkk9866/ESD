"""Build unified evidence-pool datasets: train / calib / test for HotpotQA, 2Wiki, MuSiQue, NQ."""
import gzip, json, os, random, re, string, sys
import pandas as pd
from units import split_sentences

RAW = "data/raw"
OUT = "data/proc"
os.makedirs(OUT, exist_ok=True)
N_TRAIN, N_TEST, N_CALIB = 8000, 1000, 500


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains(sent, answers):
    ns = " " + norm(sent) + " "
    return any(norm(a) and (" " + norm(a) + " ") in ns for a in answers)


def _ctx_pairs(ctx):
    if isinstance(ctx, dict):
        return list(zip(ctx["title"], ctx["sentences"]))
    return [(c[0], c[1]) for c in ctx]


def _sf_pairs(sf):
    if isinstance(sf, dict):
        return list(zip(sf["title"], sf["sent_id"]))
    return [(s[0], s[1]) for s in sf]


def from_hotpot_like(row, ds):
    sf = set((t, int(i)) for t, i in _sf_pairs(row["supporting_facts"]))
    ps = []
    for title, sents in _ctx_pairs(row["context"]):
        sents = [s.strip() for s in list(sents) if s and s.strip()]
        if not sents:
            continue
        gold = [1 if (title, i) in sf else 0 for i in range(len(sents))]
        ps.append({"title": title, "sents": sents, "gold": gold, "is_gold": int(any(gold))})
    ans = [row["answer"]]
    return {"id": str(row["id"] if "id" in row else row["_id"]), "dataset": ds, "question": row["question"], "answers": ans, "passages": ps}


def load_hotpot(split):
    d = os.path.join(RAW, "hotpotqa/distractor")
    fs = sorted(f for f in os.listdir(d) if f.startswith(split))
    df = pd.concat([pd.read_parquet(os.path.join(d, f)) for f in fs])
    return [from_hotpot_like(r, "hotpotqa") for r in df.to_dict("records")]


def load_2wiki(split):
    d = os.path.join(RAW, "2wiki")
    fs = sorted(f for f in os.listdir(d) if f.startswith(split))
    df = pd.concat([pd.read_parquet(os.path.join(d, f)) for f in fs])
    out = []
    for r in df.to_dict("records"):
        for k in ("context", "supporting_facts"):
            if isinstance(r[k], str):
                r[k] = json.loads(r[k])
        out.append(from_hotpot_like(r, "2wiki"))
    return out


def load_musique(split):
    f = os.path.join(RAW, "musique", f"musique_ans_v1.0_{split}.jsonl")
    out = []
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        answers = [r["answer"]] + list(r.get("answer_aliases", []))
        hop_ans = {}
        for h in r["question_decomposition"]:
            hop_ans.setdefault(h["paragraph_support_idx"], []).append(h["answer"])
        ps = []
        for p in r["paragraphs"]:
            sents = split_sentences(p["paragraph_text"])
            if not sents:
                continue
            gold = [0] * len(sents)
            if p["is_supporting"]:
                a = hop_ans.get(p["idx"], []) + answers
                gold = [1 if contains(s, a) else 0 for s in sents]
                if not any(gold):
                    gold = [1] * len(sents)
            ps.append({"title": p["title"], "sents": sents, "gold": gold, "is_gold": int(p["is_supporting"])})
        out.append({"id": r["id"], "dataset": "musique", "question": r["question"], "answers": answers, "passages": ps})
    return out


def load_nq():
    f = os.path.join(RAW, "nq_litm", "nq20.jsonl.gz")
    out = []
    for i, line in enumerate(gzip.open(f, "rt", encoding="utf-8")):
        r = json.loads(line)
        ps = []
        for c in r["ctxs"]:
            sents = split_sentences(c["text"])
            if not sents:
                continue
            gold = [0] * len(sents)
            if c.get("isgold"):
                gold = [1 if contains(s, r["answers"]) else 0 for s in sents]
                if not any(gold):
                    gold = [1] * len(sents)
            ps.append({"title": c["title"], "sents": sents, "gold": gold, "is_gold": int(bool(c.get("isgold")))})
        out.append({"id": f"nq{i}", "dataset": "nq", "question": r["question"], "answers": r["answers"], "passages": ps})
    return out


def shuffle_passages(exs, seed):
    for e in exs:
        rng = random.Random(f"{seed}-{e['id']}")
        rng.shuffle(e["passages"])
    return exs


def dump(exs, name):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        for e in exs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(name, len(exs), flush=True)


def main():
    rng = random.Random(13)
    train, calib, test = [], [], []
    for name, tr_fn, dev_fn in [("hotpotqa", lambda: load_hotpot("train"), lambda: load_hotpot("validation")),
                                ("2wiki", lambda: load_2wiki("train"), lambda: load_2wiki("validation")),
                                ("musique", lambda: load_musique("train"), lambda: load_musique("dev"))]:
        tr = tr_fn(); rng.shuffle(tr)
        tr = [e for e in tr if any(any(p["gold"]) for p in e["passages"])][:N_TRAIN]
        dv = dev_fn(); rng.shuffle(dv)
        dv = [e for e in dv if any(any(p["gold"]) for p in e["passages"])]
        train += tr
        dump(shuffle_passages(dv[:N_TEST], 0), f"test_{name}.jsonl")
        dump(shuffle_passages(dv[N_TEST:N_TEST + N_CALIB], 0), f"calib_{name}.jsonl")
    nq = load_nq(); rng.shuffle(nq)
    dump(shuffle_passages(nq[:N_TEST], 0), "test_nq.jsonl")
    dump(shuffle_passages(nq[N_TEST:N_TEST + N_CALIB], 0), "calib_nq.jsonl")
    rng.shuffle(train)
    dump(shuffle_passages(train, 0), "train.jsonl")


if __name__ == "__main__":
    main()
