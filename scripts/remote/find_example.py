"""Read-only: find short bridge questions where ESD answers correctly and the same-length independent scorer does not.
Prints the question, the ESD chain with pi(unit) and pi(END) at each step, and FT-CE's top units."""
import json, sys
sys.path.insert(0, "src")
from esd import units, gold_of

for ds in ["2wiki", "hotpotqa", "musique"]:
    ex = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/proc/test_{ds}.jsonl", encoding="utf-8")}
    P = {m: {json.loads(l)["id"]: json.loads(l) for l in open(f"results/preds/qwen7b/{ds}/{m}.jsonl")} for m in ["esd", "ftce_m_esd", "full", "ftce_r4"]}
    CH = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/chains/{ds}/esd.jsonl")}
    FT = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/compressed/{ds}/ftce_m_esd.jsonl", encoding="utf-8")}
    n = 0
    for i, e in ex.items():
        if not (P["esd"][i]["em"] and not P["ftce_m_esd"][i]["em"] and not P["full"][i]["em"]):
            continue
        us = units(e); g = set(gold_of(e, us)); ch = CH[i]["chain"]
        if len(ch) != 2 or {u for u, _, _ in ch} != g or len(e["question"]) > 95:
            continue
        if any(len(us[u][2]) > 130 for u, _, _ in ch):
            continue
        idx = {(a, b): k for k, (a, b, _) in enumerate(us)}
        ft = [idx[tuple(x)] for x in FT[i]["kept"]]
        print("=" * 30, ds, i, "| pool units", len(us), "| passages", len(e["passages"]))
        print("Q:", e["question"], "| gold:", e["answers"][:2])
        print("preds: esd=", P["esd"][i]["pred"], "| ftce_m_esd=", P["ftce_m_esd"][i]["pred"], "| full=", P["full"][i]["pred"], "| ftce_r4=", P["ftce_r4"][i]["pred"])
        z0 = CH[i]["z0"]; top = sorted(range(len(z0)), key=lambda k: -z0[k])[:4]
        for t, (u, pu, pe) in enumerate(ch):
            print(f"  step{t+1}: pi={pu:.2f} pEND={pe:.2f} gold={u in g} z0rank={sorted(range(len(z0)), key=lambda k:-z0[k]).index(u)+1} :: {us[u][2]}")
        print(f"  halt: pEND={CH[i]['p_end']:.2f}")
        for k in ft:
            print(f"  FT-CE kept gold={k in g}: {us[k][2][:140]}")
        n += 1
        if n >= 4:
            break
