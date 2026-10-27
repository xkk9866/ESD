"""Decode evidence chains once per question (test + calibration), then write compressed evidence for
  <name>        : Bayes stopping on sufficiency, P(chain sufficient) >= 1/2 (the model decides the length)
  <name>_unit   : ablation, Bayes stopping on the next unit alone (p < 1/2)
  <name>_s{t}   : sufficiency level t (operating curve)
  <name>_p{r}   : the same chains truncated at one corpus-level token price meeting the 1/r average budget
Chains are dumped to results/chains/<ds>/<name>.jsonl for analysis."""
import argparse, json, os, time
import numpy as np
from ecd import ChainDecoder, sufficiency_stop, unit_stop, price_stop, units
from baselines import load, render_kept, nwords_full, DATASETS


def decode_all(dec, exs, path):
    done = {}
    if os.path.exists(path):  # resume; a line cut by an interrupted run is dropped and re-decoded
        for l in open(path, encoding="utf-8"):
            try:
                r = json.loads(l); done[r["id"]] = r
            except json.JSONDecodeError:
                pass
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in done.values())
    with open(path, "a", encoding="utf-8") as f:
        for e in exs:
            if e["id"] in done:
                continue
            t0 = time.time(); p0, ch = dec.decode(e)
            r = {"id": e["id"], "p0": p0, "chain": ch, "time": time.time() - t0}
            done[e["id"]] = r; f.write(json.dumps(r) + "\n")
    return [done[e["id"]] for e in exs]


def kept_of(ex, idx):
    us = units(ex)
    return [(us[i][0], us[i][1]) for i in idx]


def write(ds, name, exs, recs, sel):
    with open(f"results/compressed/{ds}/{name}.jsonl", "w", encoding="utf-8") as f:
        for e, r in zip(exs, recs):
            kept = kept_of(e, sel(e, r))
            f.write(json.dumps({"id": e["id"], "context": render_kept(e, kept), "kept": kept, "time": r["time"]}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/ecd")
    ap.add_argument("--name", default="ecd")
    ap.add_argument("--ctx", default="chain")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--ratios", default="2,4,8,16")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--taus", default="0.3,0.7,0.9", help="sufficiency levels for the operating curve (main: 1/2)")
    a = ap.parse_args()
    a.taus = [float(t) for t in a.taus.split(",")] if a.taus else []
    dec = ChainDecoder(a.ckpt, ctx=a.ctx)
    LAMS = [0.0] + list(np.logspace(-4, np.log10(5.0), 60))
    for ds in a.datasets.split(","):
        os.makedirs(f"results/chains/{ds}", exist_ok=True); os.makedirs(f"results/compressed/{ds}", exist_ok=True)
        exs = load(ds, a.n)
        recs = decode_all(dec, exs, f"results/chains/{ds}/{a.name}.jsonl")
        write(ds, a.name, exs, recs, lambda e, r: sufficiency_stop(r["chain"]))
        write(ds, f"{a.name}_unit", exs, recs, lambda e, r: unit_stop(r["chain"]))
        for tau in a.taus:
            write(ds, f"{a.name}_s{tau}", exs, recs, lambda e, r, tau=tau: sufficiency_stop(r["chain"], tau))
        if not a.ratios:
            continue
        cal = [json.loads(l) for l in open(f"data/proc/calib_{ds}.jsonl", encoding="utf-8")]
        crec = decode_all(dec, cal, f"results/chains/{ds}/{a.name}_calib.jsonl")
        full = np.mean([nwords_full(e) for e in cal])
        mc = np.array([[len(render_kept(e, kept_of(e, price_stop(e, r["chain"], lam))).split()) for lam in LAMS]
                       for e, r in zip(cal, crec)]).mean(0)
        for rr in map(int, a.ratios.split(",")):
            ok = [lam for lam, c in zip(LAMS, mc) if c <= full / rr]
            lam = min(ok) if ok else LAMS[-1]
            print(a.name, ds, f"p{rr}", "lambda", lam, flush=True)
            write(ds, f"{a.name}_p{rr}", exs, recs, lambda e, r, lam=lam: price_stop(e, r["chain"], lam))
        b = [len(render_kept(e, kept_of(e, sufficiency_stop(r["chain"]))).split()) for e, r in zip(exs, recs)]
        print(a.name, ds, "bayes words %.1f of %.1f" % (np.mean(b), np.mean([nwords_full(e) for e in exs])), flush=True)


if __name__ == "__main__":
    main()
