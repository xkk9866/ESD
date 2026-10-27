"""Decode every test question with SPA and write compressed evidence.
  <name>        greedy ascent on V with add / remove / stay moves (the potential decides the set and its size)
  <name>_marg   same network, units ranked by their singleton potential V(q,{u}) and cut at the size SPA returned
                (the network never sees a set: isolates what reading the set contributes)
  --remove 0    climbing without remove moves (written as <name>, pass a different --name)
  --beam B      set-level beam search on V (<name>)
Paths go to results/chains/<ds>/<name>.jsonl."""
import argparse, json, os, time
import numpy as np
from spa import Climber, units
from baselines import load, render_kept, DATASETS


def write(ds, name, exs, recs, key):
    with open(f"results/compressed/{ds}/{name}.jsonl", "w", encoding="utf-8") as f:
        for e, r in zip(exs, recs):
            us = units(e); kept = [(us[i][0], us[i][1]) for i in key(r)]
            f.write(json.dumps({"id": e["id"], "context": render_kept(e, kept), "kept": kept, "time": r["time"]}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/spa")
    ap.add_argument("--name", default="spa")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--remove", type=int, default=1)
    ap.add_argument("--beam", type=int, default=1)
    a = ap.parse_args()
    cl = Climber(a.ckpt, a.max_steps, bool(a.remove), beam=a.beam)
    for ds in a.datasets.split(","):
        os.makedirs(f"results/chains/{ds}", exist_ok=True); os.makedirs(f"results/compressed/{ds}", exist_ok=True)
        exs = load(ds, a.n); recs = []
        cp = f"results/chains/{ds}/{a.name}.jsonl"
        with open(cp, "w", encoding="utf-8") as f:
            for n, e in enumerate(exs):
                t0 = time.time()
                if a.beam > 1:
                    S, v = cl.beam_climb(e); r = {"id": e["id"], "set": S, "v": v, "path": [], "halted": True}
                else:
                    z0, path, v, halted = cl.climb(e)
                    C = []
                    for kind, u, _ in path:
                        C = C + [u] if kind == "add" else [x for x in C if x != u]
                    r = {"id": e["id"], "set": sorted(C), "v": v, "path": path, "z0": z0, "halted": halted}
                r["time"] = time.time() - t0; recs.append(r); f.write(json.dumps(r) + "\n")
                if n % 200 == 0:
                    print(ds, n, len(r["set"]), "%.2fs" % r["time"], flush=True)
        write(ds, a.name, exs, recs, lambda r: r["set"])
        if a.beam == 1:
            write(ds, f"{a.name}_marg", exs, recs, lambda r: [int(j) for j in np.argsort(-np.asarray(r["z0"]))[: max(1, len(r["set"]))]])
        L = [len(r["set"]) for r in recs]
        rm = [sum(k == "remove" for k, _, _ in r["path"]) for r in recs]
        print(a.name, ds, "units %.2f removes/q %.3f q-with-remove %.1f%% unhalted %.1f%% sec %.3f" % (
            np.mean(L), np.mean(rm), 100 * np.mean([x > 0 for x in rm]), 100 * np.mean([not r["halted"] for r in recs]),
            np.mean([r["time"] for r in recs])), flush=True)


if __name__ == "__main__":
    main()
