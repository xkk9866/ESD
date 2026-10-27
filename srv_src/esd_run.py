"""Decode evidence sets once per test question and write compressed evidence:
  <name>       : ESD, greedy decoding of the learned policy until END wins (the policy decides the length)
  <name>_marg  : same network, units ranked by the empty-state logits (the policy never sees its set),
                 cut at the number of units ESD kept for that question (isolates *which* units the set leads to)
Decoded chains go to results/chains/<ds>/<name>.jsonl for analysis."""
import argparse, json, os, time
import numpy as np
from esd import SetDecoder, units
from baselines import load, render_kept, DATASETS


def decode_all(dec, exs, path):
    done = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try:
                r = json.loads(l); done[r["id"]] = r
            except json.JSONDecodeError:
                pass
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in done.values())
    with open(path, "a", encoding="utf-8") as f:
        for n, e in enumerate(exs):
            if e["id"] in done:
                continue
            t0 = time.time(); z0, ch, pend = dec.decode(e)
            r = {"id": e["id"], "z0": z0, "chain": ch, "p_end": pend, "time": time.time() - t0}
            done[e["id"]] = r; f.write(json.dumps(r) + "\n")
            if n % 200 == 0:
                print(n, len(ch), "%.2fs" % r["time"], flush=True)
    return [done[e["id"]] for e in exs]


def write(ds, name, exs, recs, sel):
    with open(f"results/compressed/{ds}/{name}.jsonl", "w", encoding="utf-8") as f:
        for e, r in zip(exs, recs):
            us = units(e); kept = [(us[i][0], us[i][1]) for i in sel(e, r)]
            f.write(json.dumps({"id": e["id"], "context": render_kept(e, kept), "kept": kept, "time": r["time"]}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/esd")
    ap.add_argument("--name", default="esd")
    ap.add_argument("--ctx", default="set")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=12)
    a = ap.parse_args()
    dec = SetDecoder(a.ckpt, max_steps=a.max_steps, ctx=a.ctx)
    for ds in a.datasets.split(","):
        os.makedirs(f"results/chains/{ds}", exist_ok=True); os.makedirs(f"results/compressed/{ds}", exist_ok=True)
        exs = load(ds, a.n)
        recs = decode_all(dec, exs, f"results/chains/{ds}/{a.name}.jsonl")
        write(ds, a.name, exs, recs, lambda e, r: [u for u, _, _ in r["chain"]])
        if a.ctx == "set":
            write(ds, f"{a.name}_marg", exs, recs, lambda e, r: [int(j) for j in np.argsort(-np.asarray(r["z0"]))[: max(1, len(r["chain"]))]])
        L = [len(r["chain"]) for r in recs]
        print(a.name, ds, "units %.2f capped %.1f%% sec %.3f" % (np.mean(L), 100 * np.mean([r["p_end"] is None for r in recs]),
              np.mean([r["time"] for r in recs])), flush=True)


if __name__ == "__main__":
    main()
