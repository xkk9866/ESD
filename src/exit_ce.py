"""EXIT-style context-aware sentence classifier trained on our labels (Hwang et al., 2025, re-implemented with the
same backbone and supervision as FT-CE / ESD). Each sentence is classified with its own full passage as context,
"[sentence] <doc> [title: passage]", and kept iff P(useful) >= 1/2 -- EXIT's decision rule. Unlike ESD, the decision is
made per sentence and never sees the other kept sentences. Writes results/compressed/<ds>/exitce.jsonl."""
import argparse, json, os, random, time
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from baselines import load, render_kept, DATASETS


def items(e):
    for pi, p in enumerate(e["passages"]):
        doc = f"{p['title']}: {' '.join(p['sents'])}"
        for si, s in enumerate(p["sents"]):
            yield pi, si, f"{p['title']}: {s} <doc> {doc}", bool(p["gold"][si])


def enc(tok, qs, xs):
    return tok(qs, xs, padding=True, truncation="only_second", max_length=384, return_tensors="pt").to("cuda")


def train(a):
    rng = random.Random(0); pairs = []
    for l in open(a.train, encoding="utf-8"):
        e = json.loads(l); it = list(items(e))
        pos = [x for x in it if x[3]]; neg = [x for x in it if not x[3]]; rng.shuffle(neg)
        pairs += [(e["question"], x[2], 1.0) for x in pos] + [(e["question"], x[2], 0.0) for x in neg[: a.neg]]
    rng.shuffle(pairs); print("pairs", len(pairs), flush=True)
    tok = AutoTokenizer.from_pretrained(a.backbone)
    m = AutoModelForSequenceClassification.from_pretrained(a.backbone).cuda()
    m.roberta.embeddings.word_embeddings.weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    steps = len(pairs) // a.bs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    m.gradient_checkpointing_enable(); m.train(); t0 = time.time()
    for i in range(steps):
        b = pairs[i * a.bs:(i + 1) * a.bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m(**enc(tok, [x[0] for x in b], [x[1] for x in b])).logits.view(-1)
        loss = F.binary_cross_entropy_with_logits(z.float(), torch.tensor([x[2] for x in b], device="cuda"))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if i % 200 == 0:
            print(f"step {i}/{steps} loss {loss.item():.4f} {(time.time()-t0)/60:.1f}min", flush=True)
    m.save_pretrained(a.out); tok.save_pretrained(a.out)


@torch.no_grad()
def infer(a):
    tok = AutoTokenizer.from_pretrained(a.out)
    m = AutoModelForSequenceClassification.from_pretrained(a.out, torch_dtype=torch.bfloat16).cuda().eval()
    for ds in a.datasets.split(","):
        os.makedirs(f"results/compressed/{ds}", exist_ok=True)
        with open(f"results/compressed/{ds}/exitce.jsonl", "w", encoding="utf-8") as f:
            for e in load(ds, a.n):
                t0 = time.time(); it = list(items(e)); z = []
                for i in range(0, len(it), 64):
                    z += m(**enc(tok, [e["question"]] * len(it[i:i + 64]), [x[2] for x in it[i:i + 64]])).logits.view(-1).float().tolist()
                kept = [(x[0], x[1]) for x, s in zip(it, z) if s >= 0]
                if not kept:
                    j = max(range(len(z)), key=lambda k: z[k]); kept = [(it[j][0], it[j][1])]
                f.write(json.dumps({"id": e["id"], "context": render_kept(e, kept), "kept": kept, "time": time.time() - t0}) + "\n")
        print("exitce", ds, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train")
    ap.add_argument("--backbone", default="models/bge-reranker-v2-m3")
    ap.add_argument("--train", default="data/proc/train.jsonl")
    ap.add_argument("--out", default="ckpt/exitce")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--neg", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    train(a) if a.mode == "train" else infer(a)


if __name__ == "__main__":
    main()
