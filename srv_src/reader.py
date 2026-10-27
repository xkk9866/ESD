"""int4 (AWQ / bnb-4bit) reader LLM with vLLM. Two modes:
  answer    : read every compressed-evidence file and answer -> results/preds/<reader>/<ds>/<method>.jsonl
  summarize : abstractive query-focused LLM compression baseline -> results/compressed/<ds>/llm_summary.jsonl"""
import argparse, glob, json, os, time
from vllm import LLM, SamplingParams
from metrics import em, f1, acc

PROMPT = ("Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\n"
          "Documents:\n{ctx}\n\nQuestion: {q}\nAnswer:")
PROMPT_CB = "Answer the question. Only give me the answer and do not output any other words.\n\nQuestion: {q}\nAnswer:"
SUMM = ("Compress the documents into a short evidence summary (at most 80 words) that keeps every fact needed to answer the "
        "question, including entity relations, negations, conditions and uncertainty. Do not answer the question.\n\n"
        "Documents:\n{ctx}\n\nQuestion: {q}\nEvidence summary:")


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def clean(ctx, q):
    ctx = ctx.strip()
    if ctx.endswith(q):
        ctx = ctx[: -len(q)].strip()
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--name", default="qwen7b")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--mode", default="answer")
    ap.add_argument("--datasets", default="nq,hotpotqa,musique,2wiki")
    ap.add_argument("--methods", default="*")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    kw = dict(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.92")), enable_prefix_caching=False, seed=0)
    if a.quant:
        kw.update(quantization=a.quant, load_format="bitsandbytes" if a.quant == "bitsandbytes" else "auto")
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    for ds in a.datasets.split(","):
        test = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")}
        if a.mode == "summarize":
            exs = list(test.values())[:1000]
            ps = [chat(tok, SUMM.format(ctx="\n".join(f"{p['title']}: {' '.join(p['sents'])}" for p in e["passages"]), q=e["question"])) for e in exs]
            t0 = time.time()
            outs = llm.generate(ps, SamplingParams(temperature=0, max_tokens=160))
            dt = (time.time() - t0) / len(exs)
            with open(f"results/compressed/{ds}/llm_summary.jsonl", "w", encoding="utf-8") as f:
                for e, o in zip(exs, outs):
                    f.write(json.dumps({"id": e["id"], "context": o.outputs[0].text.strip(), "time": dt}) + "\n")
            continue
        out_dir = f"results/preds/{a.name}/{ds}"; os.makedirs(out_dir, exist_ok=True)
        pats = a.methods.strip("{}").split(",")
        files = sorted(set(f for pt in pats for f in glob.glob(f"results/compressed/{ds}/{pt}.jsonl")))
        for fp in files:
            meth = os.path.basename(fp)[:-6]
            op = f"{out_dir}/{meth}.jsonl"
            if os.path.exists(op) and not a.overwrite:
                continue
            recs = [json.loads(l) for l in open(fp, encoding="utf-8")]
            ps, ntok = [], []
            for r in recs:
                e = test[r["id"]]
                ctx = clean(r["context"], e["question"])
                text = PROMPT_CB.format(q=e["question"]) if meth == "closedbook" else PROMPT.format(ctx=ctx, q=e["question"])
                ps.append(chat(tok, text))
                ntok.append(len(tok(ctx)["input_ids"]) if ctx else 0)
            t0 = time.time()
            outs = llm.generate(ps, SamplingParams(temperature=0, max_tokens=32))
            dt = (time.time() - t0) / len(recs)
            res = []
            with open(op, "w", encoding="utf-8") as f:
                for r, o, n in zip(recs, outs, ntok):
                    e = test[r["id"]]
                    pred = o.outputs[0].text.strip().split("\n")[0]
                    row = {"id": r["id"], "pred": pred, "em": em(pred, e["answers"]), "f1": f1(pred, e["answers"]),
                           "acc": acc(pred, e["answers"]), "ctx_tokens": n, "comp_time": r.get("time", 0.0), "read_time": dt}
                    res.append(row); f.write(json.dumps(row) + "\n")
            print(ds, meth, "EM %.2f F1 %.2f tok %.0f" % (100 * sum(x["em"] for x in res) / len(res),
                  100 * sum(x["f1"] for x in res) / len(res), sum(ntok) / len(ntok)), flush=True)


if __name__ == "__main__":
    main()
