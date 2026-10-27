"""Page / figure / table / section statistics of the reference papers in sample/."""
import glob, os, re
import fitz

for pdf in sorted(glob.glob("sample/*.pdf")):
    d = fitz.open(pdf)
    t = "\n".join(p.get_text() for p in d)
    ref = re.search(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", t)
    body = t[: ref.start()] if ref else t
    ref_page = next((i + 1 for i, p in enumerate(d) if re.search(r"\n\s*(References|REFERENCES)\s*\n", "\n" + p.get_text())), None)
    figs = sorted(set(int(x) for x in re.findall(r"\n\s*(?:Figure|Fig\.)\s*(\d+)[:.]", body)))
    tabs = sorted(set(int(x) for x in re.findall(r"\n\s*Table\s*(\d+)[:.]", body)))
    words = len(body.split())
    nref = len(re.findall(r"\n\[\d+\]|\n[A-Z][a-z]+,? [A-Z]\.", t[ref.start():])) if ref else 0
    print(f"{os.path.basename(pdf):32s} pages {d.page_count:2d} refs@p{ref_page} body_words {words:6d} figs {len(figs)} tabs {len(tabs)}")
    heads = re.findall(r"\n(\d(?:\.\d)?\.?\s+[A-Z][A-Za-z ,\-:&]{3,60})\n", body)
    print("   sections:", " | ".join(h.strip() for h in heads[:24]))
