"""AdaComp-style adaptive-length baseline on our labels (Zhang et al., 2024): an independent scorer ranks the units, and a
separate length predictor reads the question and the top-K ranked units and predicts how many of them to keep.
Ranker = FT-CE (same labels as ESD). Predictor = the same backbone as a K-way classifier, label k* = the smallest k
whose top-k contains every gold unit (capped at K). Writes results/compressed/<ds>/adacomp.jsonl."""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from esd import units, gold_of, Net
from baselines import load, render_kept, DATASETS

K = 10


def ranked(net, e):
    us = units(e)
    z = net.logits([e["question"]] * len(us), [u[2] for u in us], bs=128)
    return [int(j) for j in np.argsort(-z)], us


def text_of(e, order, us):
    return " | ".join(us[j][2] for j in order[:K])


def label(e, order):
    g = set(gold_of(e))
    for k in range(1, K + 1):
        if g <= set(order[:k]):
            return k
    return K


def train(a):
    cache = "ckpt/adacomp_train.json"
    if not os.path.exists(cache):
        net = Net(a.ranker, 384); net.m.eval(); D = []
        for n, l in enumerate(open(a.train, encoding="utf-8")):
            e = json.loads(l)
            if not gold_of(e):
                continue
            o, us = ranked(net, e)
            D.append((e["question"], text_of(e, o, us), label(e, o)))
            if n % 4000 == 0:
                print("rank", n, flush=True)
        json.dump(D, open(cache, "w")); del net; torch.cuda.empty_cache()
    D = json.load(open(cache)); rng = random.Random(0); rng.shuffle(D)
    print("label dist", np.bincount([d[2] for d in D], minlength=K + 1)[1:].tolist(), flush=True)
    tok = AutoTokenizer.from_pretrained(a.backbone)
    m = AutoModelForSequenceClassification.from_pretrained(a.backbone, num_labels=K, ignore_mismatched_sizes=True).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(D) // a.bs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    m.gradient_checkpointing_enable(); m.train(); t0 = time.time()
    for i in range(steps):
        b = D[i * a.bs:(i + 1) * a.bs]
        enc = tok([x[0] for x in b], [x[1] for x in b], padding=True, truncation="only_second", max_length=512, return_tensors="pt").to("cuda")
        y = torch.tensor([x[2] - 1 for x in b], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m(**enc).logits.float()
        loss = F.cross_entropy(z, y); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if i % 200 == 0:
            print(f"step {i}/{steps} loss {loss.item():.3f} {(time.time()-t0)/60:.1f}min", flush=True)
    m.save_pretrained(a.out); tok.save_pretrained(a.out)


@torch.no_grad()
def infer(a):
    net = Net(a.ranker, 384); net.m.eval()
    tok = AutoTokenizer.from_pretrained(a.out)
    m = AutoModelForSequenceClassification.from_pretrained(a.out, torch_dtype=torch.bfloat16).cuda().eval()
    for ds in a.datasets.split(","):
        os.makedirs(f"results/compressed/{ds}", exist_ok=True)
        with open(f"results/compressed/{ds}/adacomp.jsonl", "w", encoding="utf-8") as f:
            for e in load(ds, a.n):
                t0 = time.time(); o, us = ranked(net, e)
                enc = tok([e["question"]], [text_of(e, o, us)], truncation="only_second", max_length=512, return_tensors="pt").to("cuda")
                k = int(m(**enc).logits.argmax(-1)) + 1
                kept = [(us[j][0], us[j][1]) for j in o[:k]]
                f.write(json.dumps({"id": e["id"], "context": render_kept(e, kept), "kept": kept, "time": time.time() - t0}) + "\n")
        print("adacomp", ds, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train")
    ap.add_argument("--ranker", default="ckpt/ftce")
    ap.add_argument("--backbone", default="ckpt/ftce")
    ap.add_argument("--train", default="data/proc/train.jsonl")
    ap.add_argument("--out", default="ckpt/adacomp")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    train(a) if a.mode == "train" else infer(a)


if __name__ == "__main__":
    main()
