"""Evidence units, fidelity-dependency edges, entity links and qualifier cues."""
import re

_ABBR = r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bJr)(?<!\bSr)(?<!\bvs)(?<!\bNo)(?<!\bU\.S)(?<!\binc)(?<!\bInc)(?<!\bCo)(?<!\bLtd)"
_SPLIT = re.compile(_ABBR + r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = [s.strip() for s in _SPLIT.split(text) if s.strip()]
    # merge very short fragments into previous sentence
    out = []
    for s in sents:
        if out and len(s.split()) < 3:
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out


# ---------------- qualifier cues (negation / modality / condition / quantity) ----------------
NEG = {"not", "no", "never", "none", "neither", "nor", "without", "cannot", "n't", "nobody", "nothing", "nowhere", "lack", "lacks", "denied", "denies"}
HEDGE = {"may", "might", "could", "possibly", "probably", "reportedly", "allegedly", "likely", "unlikely", "approximately",
         "about", "around", "estimated", "roughly", "nearly", "almost", "unconfirmed", "disputed", "claimed", "claims",
         "believed", "suggested", "uncertain", "unclear", "perhaps", "some", "partly", "partially", "mostly"}
COND = {"if", "unless", "only", "except", "until", "provided", "whereas", "although", "though", "before", "after",
        "since", "while", "during", "between", "first", "last", "former", "later", "earlier", "originally"}
_NUM = re.compile(r"\b\d[\d,.]*\b")
_TOK = re.compile(r"[A-Za-z]+(?:'t)?|\d[\d,.]*")


def qualifier_cues(sent):
    toks = [t.lower() for t in _TOK.findall(sent)]
    cues = []
    for t in toks:
        if t.endswith("n't"):
            cues.append("n't")
        elif t in NEG or t in HEDGE or t in COND:
            cues.append(t)
    cues += _NUM.findall(sent)
    return cues


# ---------------- fidelity dependency constraints ----------------
_ANAPH = re.compile(r"^(?:\W*)(He|She|It|They|His|Her|Its|Their|Him|Them|This|These|That|Those|Such|The latter|The former|"
                    r"The (?:company|band|film|album|song|team|club|city|town|village|book|novel|series|show|game|group|"
                    r"station|school|university|party|river|building|church|species|player|singer|actor|actress|"
                    r"author|writer|director|magazine|newspaper|airport|ship|episode|county|district|organization|"
                    r"region|country|state|province|area|work|church|museum|park))\b")
_CONTRAST = re.compile(r"^(?:\W*)(However|But|Although|Though|Nevertheless|Nonetheless|Yet|Still|Instead|In contrast|"
                       r"Despite|Critics|Others|Some|Later|Subsequently|Originally|Previously|Prior|Unlike|Except|"
                       r"Only|Not|Neither|Nor|It was later|It was not)\b")


def dependency_edges(sents):
    """Return list of (u, v) local sentence indices meaning keep(u) => keep(v)."""
    edges = []
    for i, s in enumerate(sents):
        if i == 0:
            continue
        if _ANAPH.match(s):
            edges.append((i, i - 1))
        if _CONTRAST.match(s):
            # a qualifying/contrasting sentence and the claim it qualifies are mutually dependent:
            # dropping the qualifier while keeping the claim would strengthen the evidence.
            edges.append((i, i - 1))
            edges.append((i - 1, i))
    return edges


_ENT = re.compile(r"\b(?:[A-Z][\w'&.-]+(?:\s+(?:of|the|de|la|and|von|van|du|del)?\s*[A-Z][\w'&.-]+)*)|\b1[0-9]{3}\b|\b20[0-9]{2}\b")
_STOPENT = {"The", "A", "An", "He", "She", "It", "They", "His", "Her", "In", "On", "At", "This", "That", "These", "Those",
            "However", "But", "After", "Before", "During", "When", "While", "Its", "Their", "There", "As", "For", "From",
            "With", "By", "Of", "And", "Or", "Although", "Since", "Following", "According", "Some", "Many", "Most"}


def entities(text):
    ents = set()
    for m in _ENT.findall(text):
        m = m.strip()
        if m in _STOPENT or len(m) < 3:
            continue
        ents.add(m.lower())
    return ents
