import json, sys
S = json.load(open("results/live_summary.json"))
pre = sys.argv[1].split(",") if len(sys.argv) > 1 else None
for r in S:
    for ds in S[r]:
        print("##", r, ds)
        for m, x in sorted(S[r][ds].items()):
            if pre and not any(m == p or (p.endswith("*") and m.startswith(p[:-1])) for p in pre):
                continue
            print(f"  {m:22s} EM {x['EM']:5.1f} F1 {x['F1']:5.1f} Acc {x['Acc']:5.1f} Tok {x['Tok']:6.0f} N {x['N']:4d} ms {x['CompMs']:.0f}")
