"""pgfplots figure: EM vs. reader tokens for ratio vs. price allocation (Qwen2.5-7B int4)."""
import json

S = json.load(open("results/summary_qwen7b.json"))
SERIES = [("\\ours{} price", "vc3_p", "blue!75!black, mark=*", True), ("\\ours{} ratio", "vc3_r", "blue!60, dashed, mark=o", True),
          ("FT-CE price", "ftce_p", "orange!85!black, mark=square*", True), ("FT-CE ratio", "ftce_r", "orange!80, dashed, mark=square", True)]
POINTS = [("Provence", "provence", "red, mark=triangle*"), ("LongLLMLingua", "longllmlingua", "teal, mark=diamond*"),
          ("Full", "full", "black, mark=x")]
out = ["\\begin{figure}[t]", "\\centering", "\\begin{tikzpicture}", "\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.1cm},"
       " width=0.56\\columnwidth, height=4.0cm, xmode=log, xlabel={reader tokens}, ylabel style={yshift=-4pt},"
       " tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, title style={font=\\small, yshift=-4pt}, grid=major,"
       " grid style={gray!20}]"]
for gi, (ds, title) in enumerate([("2wiki", "2Wiki"), ("musique", "MuSiQue")]):
    opts = f"title={{{title}}}" + (", ylabel={EM}" if gi == 0 else "")
    if gi == 1:
        opts += (", legend style={font=\\tiny, at={(-0.15,-0.42)}, anchor=north, legend columns=4, draw=none, fill=none,"
                 " /tikz/every even column/.append style={column sep=3pt}}, legend cell align=left")
    out.append(f"\\nextgroupplot[{opts}]")
    for name, pre, style, _ in SERIES:
        pts = [(S[ds][f"{pre}{r}"]["Tok"], S[ds][f"{pre}{r}"]["EM"]) for r in (8, 4, 2) if f"{pre}{r}" in S[ds]]
        out.append(f"\\addplot[{style}, thick] coordinates {{" + " ".join(f"({t:.0f},{e:.1f})" for t, e in pts) + "};")
        if gi == 1:
            out.append(f"\\addlegendentry{{{name}}}")
    for name, m, style in POINTS:
        if m in S[ds]:
            out.append(f"\\addplot[{style}, only marks, mark size=2.4pt] coordinates {{({S[ds][m]['Tok']:.0f},{S[ds][m]['EM']:.1f})}};")
            if gi == 1:
                out.append(f"\\addlegendentry{{{name}}}")
out += ["\\end{groupplot}", "\\end{tikzpicture}",
        "\\caption{Answer quality vs.\\ reader tokens. At the same average budget, one corpus-level token price (solid) dominates a "
        "per-query ratio (dashed) for calibrated scorers, most strongly under tight budgets.}", "\\label{fig:pareto}", "\\end{figure}"]
open("paper/figures/pareto.tex", "w").write("\n".join(out))
print("ok")
