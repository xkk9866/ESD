"""Open-retrieval setting: no gold passage is guaranteed and pools are long.
Corpus = every distinct passage of the dataset's train / calibration / test pools; each test question retrieves its
top-k passages with BM25 (title + text). Gold sentence flags are kept only for retrieved passages that are gold for that
question. Writes data/proc/test_<ds>_open.jsonl and prints pool length and gold-passage recall."""
import argparse, json, re
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

TOK = re.compile(r"\w+")
STOP = set("the a an of in on at to for and or is was were be by with as from that this which who what when where how did does do".split())


def toks(t):
    return [w for w in TOK.findall(t.lower()) if w not in STOP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="hotpotqa,musique")
    ap.add_argument("--k", type=int, default=50)
    a = ap.parse_args()
    for ds in a.datasets.split(","):
        key, docs = {}, []
        files = [f"data/proc/test_{ds}.jsonl", f"data/proc/calib_{ds}.jsonl", "data/proc/train.jsonl"]
        for fp in files:
            for l in open(fp, encoding="utf-8"):
                e = json.loads(l)
                if e.get("dataset", ds) != ds:
                    continue
                for p in e["passages"]:
                    kk = (p["title"], p["sents"][0] if p["sents"] else "")
                    if kk not in key:
                        key[kk] = len(docs); docs.append({"title": p["title"], "sents": p["sents"]})
        vocab, rows, cols, vals, dl = {}, [], [], [], []
        for i, d in enumerate(docs):
            c = Counter(toks(d["title"] + " " + " ".join(d["sents"])))
            dl.append(sum(c.values()))
            for w, n in c.items():
                rows.append(i); cols.append(vocab.setdefault(w, len(vocab))); vals.append(n)
        N, dl = len(docs), np.asarray(dl, dtype=np.float32); avg = dl.mean()
        tf = csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(N, len(vocab)))
        df = np.bincount(np.asarray(cols), minlength=len(vocab))
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        k1, b = 0.9, 0.4  # standard BM25 defaults (Anserini)
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / avg)) * idf[tf.col]
        W = csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).tocsc()
        out, rec, L = [], [], []
        for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8"):
            e = json.loads(l)
            q = [vocab[t] for t in toks(e["question"]) if t in vocab]
            s = np.asarray(W[:, q].sum(1)).ravel() if q else np.zeros(N)
            top = np.argsort(-s)[: a.k]
            gold = {(p["title"], p["sents"][0] if p["sents"] else ""): p["gold"] for p in e["passages"] if any(p["gold"])}
            ps = []
            for j in top:
                d = docs[j]; kk = (d["title"], d["sents"][0] if d["sents"] else "")
                ps.append({"title": d["title"], "sents": d["sents"], "gold": gold.get(kk, [False] * len(d["sents"]))})
            found = sum(1 for p in ps if any(p["gold"]))
            rec.append(found / max(1, len(gold))); L.append(sum(len(s_.split()) for p in ps for s_ in p["sents"]))
            out.append(dict(e, passages=ps))
        with open(f"data/proc/test_{ds}_open.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e) + "\n" for e in out)
        print(ds, "corpus", N, "pool words %.0f" % np.mean(L), "gold-passage recall %.3f" % np.mean(rec),
              "all-gold %.3f" % np.mean([r == 1 for r in rec]), flush=True)


if __name__ == "__main__":
    main()
