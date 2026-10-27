"""Reader-defined sufficient sets: supervision for ESD from (question, answer) pairs only, no supporting-fact labels.
A set is sufficient when the reader answers correctly from it. For each training question, the set is grown greedily
over the M units ranked highest by a scorer: each step adds the unit that most increases the reader's log-likelihood of
the gold answer, until the reader's greedy answer is correct (at most T steps; questions never answered are dropped).
The resulting sets replace the gold flags -> data/proc/train_reader.jsonl. Also writes the SF-labelled training file
restricted to the same multi-hop questions (data/proc/train_sfsub.jsonl) for a like-for-like comparison."""
import argparse, json, os, random, sys
import numpy as np
from vllm import LLM, SamplingParams
from metrics import em, acc
from reader import PROMPT, chat
from baselines import render_kept


def nq_train():
    sys.path.insert(0, "src")
    from data_prep import load_nq
    used = {json.loads(l)["id"] for f in ["test_nq", "calib_nq"] for l in open(f"data/proc/{f}.jsonl", encoding="utf-8")}
    return [e for e in load_nq() if e["id"] not in used]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--ranker", default="ckpt/ftce")
    ap.add_argument("--per_ds", type=int, default=2000)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--T", type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(0)
    tr = [json.loads(l) for l in open("data/proc/train.jsonl", encoding="utf-8")]
    sub = []
    for ds in ["hotpotqa", "2wiki", "musique"]:
        pool = [e for e in tr if e["dataset"] == ds]; sub += rng.sample(pool, a.per_ds)
    with open("data/proc/train_sfsub.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e) + "\n" for e in sub)
    exs = sub + nq_train()
    print("questions", len(exs), flush=True)
    # candidate units: top-M of the ranker (independent scorer), computed before the LLM is loaded
    import torch
    from esd import Net, units
    net = Net(a.ranker, 384); net.m.eval(); cand = []
    for e in exs:
        us = units(e); z = net.logits([e["question"]] * len(us), [u[2] for u in us], bs=128)
        cand.append([int(j) for j in np.argsort(-z)[: a.M]])
    del net; torch.cuda.empty_cache()
    llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.9")), seed=0)
    tok = llm.get_tokenizer()
    U = [units(e) for e in exs]

    def ctx(i, S):
        return render_kept(exs[i], [(U[i][j][0], U[i][j][1]) for j in S])

    def prompt(i, S):
        return chat(tok, PROMPT.format(ctx=ctx(i, S), q=exs[i]["question"]))

    C = [[] for _ in exs]; done = [False] * len(exs)
    lp_params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    for t in range(a.T):
        jobs, ps = [], []
        for i in range(len(exs)):
            if done[i]:
                continue
            ans = exs[i]["answers"][0]
            for j in cand[i]:
                if j in C[i]:
                    continue
                p = prompt(i, sorted(C[i] + [j]))
                jobs.append((i, j, len(tok(p)["input_ids"]))); ps.append(p + ans)
        outs = llm.generate(ps, lp_params)
        best = {}
        for (i, j, n0), o in zip(jobs, outs):
            lp = sum(next(iter(d.values())).logprob for d in o.prompt_logprobs[n0:] if d)
            if i not in best or lp > best[i][1]:
                best[i] = (j, lp)
        for i, (j, _) in best.items():
            C[i].append(j)
        idx = list(best)
        gens = llm.generate([prompt(i, sorted(C[i])) for i in idx], SamplingParams(temperature=0, max_tokens=32))
        for i, g in zip(idx, gens):
            pred = g.outputs[0].text.strip().split("\n")[0]
            if em(pred, exs[i]["answers"]) or acc(pred, exs[i]["answers"]):
                done[i] = True
        print("step", t + 1, "sufficient", sum(done), "/", len(exs), flush=True)
    n = 0
    with open("data/proc/train_reader.jsonl", "w", encoding="utf-8") as f:
        for i, e in enumerate(exs):
            if not done[i]:
                continue
            S = {(U[i][j][0], U[i][j][1]) for j in C[i]}
            ps = [dict(p, gold=[(pi, si) in S for si in range(len(p["sents"]))]) for pi, p in enumerate(e["passages"])]
            f.write(json.dumps(dict(e, passages=ps)) + "\n"); n += 1
    sizes = [len(C[i]) for i in range(len(exs)) if done[i]]
    print("labelled", n, "mean set size %.2f" % np.mean(sizes), np.bincount(sizes).tolist(), flush=True)
    # agreement with SF labels on the multi-hop part
    pr, rc = [], []
    for i, e in enumerate(exs[: len(sub)]):
        if done[i]:
            g = {(pi, si) for pi, p in enumerate(e["passages"]) for si, x in enumerate(p["gold"]) if x}
            S = {(U[i][j][0], U[i][j][1]) for j in C[i]}
            pr.append(len(g & S) / len(S)); rc.append(len(g & S) / max(1, len(g)))
    print("reader sets vs SF: precision %.3f recall %.3f" % (np.mean(pr), np.mean(rc)), flush=True)


if __name__ == "__main__":
    main()
