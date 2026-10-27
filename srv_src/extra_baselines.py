"""Recent official compressors, run with their published checkpoints and prompts (every LLM int4-quantized):
  recomp_abs : RECOMP abstractive (T5-large; hotpotqa_abstractive for multi-hop sets, nq_abstractive_compressor for NQ),
               input format of the RECOMP training data, top-5 passages
  exit       : EXIT (Hwang et al., 2025): Gemma-2B-it + official LoRA, per-sentence Yes/No with its own document as
               context, keep P(Yes) >= 0.5
  compact    : CompAct (Yoon et al., 2024): Mistral-7B compressor that summarises 5 passages per iteration,
               evaluates its own summary and stops at [COMPLETE] (at most 6 iterations); if the official weights are
               unavailable, the same prompts and loop run on the int4 Qwen2.5-7B (compact_qwen)
Passages are fed in the order of a zero-shot BGE-reranker passage score (max over its sentences), standing in for the
retrieval rank these methods expect (our pools are shuffled)."""
import argparse, json, os, re, time
import numpy as np, torch
from baselines import load, DATASETS, CrossEncScorer, sent_list, M


def passage_order(scorer, e):
    sc = scorer(e); sl = sent_list(e); best = {}
    for (pi, si, _), s in zip(sl, sc):
        best[pi] = max(best.get(pi, -1e9), s)
    return sorted(best, key=lambda p: -best[p])


def write(ds, name, rows):
    with open(f"results/compressed/{ds}/{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def recomp_abs(a, ds, exs, orders):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    path = "fangyuan/nq_abstractive_compressor" if ds.startswith("nq") else "fangyuan/hotpotqa_abstractive"
    tok = AutoTokenizer.from_pretrained(path); m = AutoModelForSeq2SeqLM.from_pretrained(path).cuda().eval()
    rows = []
    for e, o in zip(exs, orders):
        t0 = time.time()
        doc = "\n".join(f"{e['passages'][p]['title']} {' '.join(e['passages'][p]['sents'])}" for p in o[:5])
        x = tok([f"Question: {e['question']}\n Document: {doc}\n Summary: "], truncation=True, max_length=1024, return_tensors="pt").to("cuda")
        with torch.no_grad():
            y = m.generate(**x, max_length=512)
        rows.append({"id": e["id"], "context": tok.decode(y[0], skip_special_tokens=True).strip(), "time": time.time() - t0})
    write(ds, "recomp_abs", rows)
    del m; torch.cuda.empty_cache()


def exit_(a, ds, exs, model):
    tok, m, yes, no = model
    rows = []
    for e in exs:
        t0 = time.time(); keep = []
        for p in e["passages"]:
            ctx = f"{p['title']}\n{' '.join(p['sents'])}"
            ps = [f"<start_of_turn>user\nQuery:\n{e['question']}\nFull context:\n{ctx}\nSentence:\n{s}\n"
                  f"Is this sentence useful in answering the query? Answer only \"Yes\" or \"No\".<end_of_turn>\n<start_of_turn>model\n" for s in p["sents"]]
            sc = []
            for i in range(0, len(ps), 16):
                b = tok(ps[i:i + 16], return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    z = m(**b).logits[:, -1, :].float()
                sc += torch.softmax(z[:, [yes, no]], -1)[:, 0].tolist()
            sel = [s for s, v in zip(p["sents"], sc) if v >= 0.5]
            if sel:
                keep.append(f"{p['title']}: {' '.join(sel)}")
        rows.append({"id": e["id"], "context": "\n".join(keep), "time": time.time() - t0})
    write(ds, "exit", rows)


def load_exit():
    """official LoRA merged into the base weights (W += B A * alpha / r), then loaded with 4-bit weights"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    merged = f"{M}/exit-gemma-2b-merged"
    if not os.path.exists(f"{merged}/config.json"):
        from safetensors.torch import load_file
        base = AutoModelForCausalLM.from_pretrained(f"{M}/gemma-2b-it", torch_dtype=torch.float32)
        cfg = json.load(open(f"{M}/exit-gemma-2b/adapter_config.json")); sc = cfg["lora_alpha"] / cfg["r"]
        ad = load_file(f"{M}/exit-gemma-2b/adapter_model.safetensors"); P = dict(base.named_parameters()); n = 0
        for k, A in ad.items():
            if ".lora_A." not in k:
                continue
            B = ad[k.replace(".lora_A.", ".lora_B.")]
            tgt = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight").replace(".lora_A.default.weight", ".weight")
            with torch.no_grad():
                P[tgt] += (B.float() @ A.float()) * sc
            n += 1
        print("merged LoRA matrices", n, flush=True)
        base.to(torch.bfloat16).save_pretrained(merged); AutoTokenizer.from_pretrained(f"{M}/gemma-2b-it").save_pretrained(merged)
        del base
    tok = AutoTokenizer.from_pretrained(merged); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(merged, device_map="cuda",
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    return tok, m, tok.convert_tokens_to_ids("Yes"), tok.convert_tokens_to_ids("No")


I0 = ("1. Generate a summary of source documents to answer the question. Ensure the summary is under 200 words and does not include any pronouns. "
      "DO NOT make assumptions or attempt to answer the question; your job is to summarize only.\n\n2. Evaluate the summary based solely on the information of it, "
      "without any additional background context: if it lacks sufficient details to answer the question, print '[INCOMPLETE]'. If it provides all necessary details, "
      "print '[COMPLETE]'. You should provide the reason of evalution.")
I1 = ("1. Generate a summary of the previous summary and the source documents to answer the question based on the evaluation of the previous summary. "
      "The evaluation indicates the missing information needed to answer the question. Ensure the summary is under 200 words and does not include any pronouns. "
      "DO NOT make assumptions or attempt to answer the question; your job is to summarize only.\n\n2. Evaluate the summary based solely on the information of it, "
      "without any additional background context: if it lacks sufficient details to answer the question, print '[INCOMPLETE]'. If it provides all necessary details, "
      "print '[COMPLETE]'. You should provide the reason of evalution.")


def parse(text):
    s = re.search(r"(Summary:)(.*?)(?=Evaluation:|$)", text, re.DOTALL) or re.search(r"(^)(.*?)(?=Evaluation:|$)", text, re.DOTALL)
    ev = re.search(r"(Evaluation:)(.*?)(?=Summary:|$)", text, re.DOTALL)
    return (s.group(2).strip().replace("\n\n", "") if s else ""), (ev.group(2).strip().replace("\n\n", "") if ev else "")


def compact(a, ds, exs, orders, llm, name="compact"):
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    st = [{"summ": "", "eval": "", "done": False, "t": 0.0} for _ in exs]
    for it in range(6):
        idx, ps = [], []
        for i, (e, o) in enumerate(zip(exs, orders)):
            seg = o[5 * it: 5 * it + 5]
            if st[i]["done"] or not seg:
                continue
            doc = "\n".join(f"{e['passages'][p]['title']} {' '.join(e['passages'][p]['sents'])}" for p in seg)
            if it == 0:
                pr = f"{I0}\n\nQuestion: {e['question']}\n\nSource documents: {doc}\n\nSummary:"
            else:
                pr = (f"{I1}\n\nQuestion: {e['question']}\n\nPrevious summary: {st[i]['summ']}\n\nEvaluation of previous summary: "
                      f"{st[i]['eval'].replace('[INCOMPLETE]', '').strip()}\n\nSource documents: {doc}\n\nSummary:")
            idx.append(i); ps.append(tok.apply_chat_template([{"role": "user", "content": pr}], tokenize=False, add_generation_prompt=True))
        if not ps:
            break
        t0 = time.time()
        outs = llm.generate(ps, SamplingParams(temperature=0, max_tokens=900))
        dt = (time.time() - t0) / len(ps)
        for i, o in zip(idx, outs):
            s, ev = parse(o.outputs[0].text)
            st[i].update(summ=s, eval=ev, t=st[i]["t"] + dt, done="[COMPLETE]" in ev)
        print("compact", ds, "iter", it, "active", len(ps), flush=True)
    write(ds, name, [{"id": e["id"], "context": s["summ"], "time": s["t"]} for e, s in zip(exs, st)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    dss = a.datasets.split(",")
    data = {ds: load(ds, a.n) for ds in dss}
    orders = {}
    if a.method in ("recomp_abs", "compact"):
        sc = CrossEncScorer(f"{M}/bge-reranker-v2-m3")
        orders = {ds: [passage_order(sc, e) for e in data[ds]] for ds in dss}
        del sc; torch.cuda.empty_cache()
    if a.method == "exit":
        model = load_exit()
    if a.method == "compact":
        from vllm import LLM
        official = os.path.exists(f"{M}/CompAct-7b/model-00003-of-00003.safetensors")
        kw = dict(quantization="bitsandbytes", load_format="bitsandbytes") if official else {}
        llm = LLM(model=f"{M}/CompAct-7b" if official else f"{M}/Qwen2.5-7B-Instruct-AWQ", max_model_len=8192,
                  gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.9")), seed=0, **kw)
        cname = "compact" if official else "compact_qwen"  # fallback: CompAct's prompt on the int4 Qwen2.5-7B
    for ds in dss:
        os.makedirs(f"results/compressed/{ds}", exist_ok=True)
        if a.method == "recomp_abs":
            recomp_abs(a, ds, data[ds], orders[ds])
        elif a.method == "exit":
            exit_(a, ds, data[ds], model)
        elif a.method == "compact":
            compact(a, ds, data[ds], orders[ds], llm, cname)
        print(a.method, ds, "done", flush=True)


if __name__ == "__main__":
    main()
