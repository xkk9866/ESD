import json, sys
S = json.load(open(f"results/summary_{sys.argv[1]}.json"))
pre = sys.argv[2].split(",") if len(sys.argv) > 2 else None
for d in S:
    print("##", d)
    for m, r in sorted(S[d].items()):
        if pre is None:
            continue
        if any(m.startswith(p) for p in pre):
            f = lambda v, k=1: "-" if v is None else f"{v:.{k}f}"
            print(f"{m:22s} EM {f(r['EM'])} F1 {f(r['F1'])} Acc {f(r.get('Acc'))} Tok {f(r['Tok'],0)} SF {f(r.get('SF'))} Ans {f(r.get('AnsRec'))}")
    if pre is None:
        print(" ".join(sorted(S[d])))
