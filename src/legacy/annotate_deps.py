"""Distil contextual-dependency labels from an int4 LLM.

For a unit A and an adjacent unit B of the same passage, the LLM decides whether A can be read correctly without B:
  1 = reference   (B is needed to know who/what A refers to)
  2 = qualification (B negates, limits, conditions or expresses uncertainty about A)
  3 = independent
Pairs: every gold unit with its previous/next unit; for training also an equal number of random non-gold units.
Output: data/proc/deps_<split>_<annotator>.jsonl with {id, pairs: [[pid, a_sid, b_sid, label], ...]}"""
import argparse, json, os, random
from vllm import LLM, SamplingParams

PROMPT = ("Passage: {title}\nSentence A: {a}\nSentence B: {b}\n\n"
          "Sentence A will be shown to a reader as evidence, possibly without Sentence B. Which option is correct?\n"
          "1. Sentence B is needed to understand who or what Sentence A refers to.\n"
          "2. Sentence B negates, qualifies, limits, or adds a condition or uncertainty to Sentence A.\n"
          "3. Sentence A is understood correctly without Sentence B.\n"
          "Answer with the number only.")


def pairs_of(ex, rng, with_negatives):
    out = []
    for pi, p in enumerate(ex["passages"]):
        n = len(p["sents"])
        srcs = [i for i in range(n) if p["gold"][i]]
        if with_negatives:
            neg = [i for i in range(n) if not p["gold"][i]]
            rng.shuffle(neg)
            srcs += neg[: max(1, len(srcs))] if neg else []
        for a in srcs:
            for b in (a - 1, a + 1):
                if 0 <= b < n:
                    out.append((pi, a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-14B-Instruct-AWQ")
    ap.add_argument("--tag", default="qwen14b")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--splits", default="train:6000,test_nq,test_hotpotqa,test_musique,test_2wiki")
    a = ap.parse_args()
    kw = dict(model=a.model, max_model_len=2048, gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.9")), seed=0)
    if a.quant:
        kw.update(quantization=a.quant, load_format="bitsandbytes")
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=2)
    for spec in a.splits.split(","):
        name, n = (spec.split(":") + [None])[:2]
        rows = [json.loads(l) for l in open(f"data/proc/{name}.jsonl", encoding="utf-8")]
        rows = rows[: int(n)] if n else rows
        rng = random.Random(0)
        jobs, prompts = [], []
        for r in rows:
            for pi, s_a, s_b in pairs_of(r, rng, with_negatives=name.startswith("train")):
                p = r["passages"][pi]
                jobs.append((r["id"], pi, s_a, s_b))
                prompts.append(tok.apply_chat_template([{"role": "user", "content": PROMPT.format(
                    title=p["title"], a=p["sents"][s_a], b=p["sents"][s_b])}], tokenize=False, add_generation_prompt=True))
        outs = llm.generate(prompts, sp)
        by = {}
        for (rid, pi, s_a, s_b), o in zip(jobs, outs):
            t = o.outputs[0].text.strip()
            lab = int(t[0]) if t[:1] in ("1", "2", "3") else 3
            by.setdefault(rid, []).append([pi, s_a, s_b, lab])
        with open(f"data/proc/deps_{name}_{a.tag}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "pairs": by.get(r["id"], [])}) + "\n")
        labs = [x[3] for v in by.values() for x in v]
        print(name, len(labs), {k: labs.count(k) for k in (1, 2, 3)}, flush=True)


if __name__ == "__main__":
    main()
