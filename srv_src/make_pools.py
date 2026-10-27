"""Pool-size robustness sets: the same test questions with k retrieved passages (all gold passages + random
distractor passages of the original pool, original order kept) -> data/proc/test_<ds>_k<k>.jsonl"""
import json, random

for ds, ks in [("musique", [5, 10]), ("nq", [5, 10])]:
    E = [json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")]
    for k in ks:
        rng = random.Random(k)
        with open(f"data/proc/test_{ds}_k{k}.jsonl", "w", encoding="utf-8") as f:
            for e in E:
                g = [i for i, p in enumerate(e["passages"]) if any(p["gold"])]
                d = [i for i in range(len(e["passages"])) if i not in g]
                keep = sorted(g + rng.sample(d, max(0, min(len(d), k - len(g)))))
                f.write(json.dumps(dict(e, passages=[e["passages"][i] for i in keep])) + "\n")
        print(ds, k, len(E))
