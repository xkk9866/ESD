"""Strong supervised baseline FT-CE: fine-tune bge-reranker-v2-m3 as an independent sentence scorer
on exactly the same supporting-sentence labels used by VeriComp (no joint encoding, no constraints, no verification)."""
import argparse, json, os, random, time
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="models/bge-reranker-v2-m3")
    ap.add_argument("--train", default="data/proc/train.jsonl")
    ap.add_argument("--out", default="ckpt/ftce")
    ap.add_argument("--neg", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    pairs = []
    for l in open(a.train, encoding="utf-8"):
        e = json.loads(l)
        pos, neg = [], []
        for p in e["passages"]:
            for s, g in zip(p["sents"], p["gold"]):
                (pos if g else neg).append(f"{p['title']}: {s}")
        rng.shuffle(neg)
        pairs += [(e["question"], s, 1.0) for s in pos] + [(e["question"], s, 0.0) for s in neg[: a.neg]]
    rng.shuffle(pairs)
    print("pairs", len(pairs), flush=True)
    tok = AutoTokenizer.from_pretrained(a.backbone)
    m = AutoModelForSequenceClassification.from_pretrained(a.backbone).cuda()
    m.roberta.embeddings.word_embeddings.weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    steps = len(pairs) // a.bs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    m.train(); t0 = time.time()
    for i in range(steps):
        b = pairs[i * a.bs:(i + 1) * a.bs]
        enc = tok([[q, s] for q, s, _ in b], padding=True, truncation=True, max_length=192, return_tensors="pt").to("cuda")
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


if __name__ == "__main__":
    main()
