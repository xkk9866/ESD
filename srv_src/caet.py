"""CAET: Constraint-Aware Evidence Transformer (the single network inside VeriComp).

One XLM-R backbone, three roles:
  * local reading  (lower L-K layers, block-diagonal: query + own passage)
  * relational reasoning (upper K layers, global attention + evidence-relation attention bias)
  * self-verification (the same upper K layers re-read the evidence under the selection mask)
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer
from units import dependency_edges, entities

# relation vocabulary between evidence units (unit 0 = query)
R_NONE, R_SELF, R_ADJ, R_PASS, R_ENT, R_DEP, R_QU, R_QQ = range(8)
N_REL = 8


class Featurizer:
    def __init__(self, tok, max_len=4096, max_q=64, max_sent=96, structural=False):
        self.tok, self.max_len, self.max_q, self.max_sent = tok, max_len, max_q, max_sent
        self.structural = structural

    def __call__(self, ex):
        tok = self.tok
        q = tok(ex["question"], add_special_tokens=False)["input_ids"][: self.max_q]
        ids = [tok.cls_token_id] + q + [tok.sep_token_id]
        uid = [0] * len(ids)
        units = [dict(kind="query", pid=-1, sid=-1, gold=0, nwords=0)]
        flat = []
        for pi, p in enumerate(ex["passages"]):
            flat.append((pi, -1, p["title"] + ":"))
            for si, s in enumerate(p["sents"]):
                flat.append((pi, si, " " + s))
        enc = tok([t for _, _, t in flat], add_special_tokens=False)["input_ids"]
        budget = self.max_len - len(ids)
        total = sum(min(len(e), self.max_sent) + (1 if si == -1 else 0) for (_, si, _), e in zip(flat, enc))
        cap = self.max_sent
        while total > budget and cap > 16:  # shrink per-sentence cap until it fits
            cap = int(cap * 0.85)
            total = sum(min(len(e), cap) + (1 if si == -1 else 0) for (_, si, _), e in zip(flat, enc))
        for (pi, si, text), e in zip(flat, enc):
            e = e[:cap]
            if len(ids) + len(e) + 1 > self.max_len:
                break
            if si == -1 and pi > 0:
                ids.append(tok.sep_token_id); uid.append(len(units) - 1)
            u = len(units)
            p = ex["passages"][pi]
            units.append(dict(kind="title" if si == -1 else "sent", pid=pi, sid=si,
                              gold=0 if si == -1 else int(p["gold"][si]),
                              nwords=len(text.split())))
            ids += e; uid += [u] * len(e)
        U = len(units)
        R = self.struct_relations(units) if self.structural else self.relations(ex, units)
        pid = torch.tensor([units[u]["pid"] for u in uid])
        return dict(input_ids=torch.tensor(ids), uid=torch.tensor(uid), pid=pid, units=units, R=R, n_units=U)

    @staticmethod
    def struct_relations(units):
        """purely structural relations (no lexical rules): self / adjacent / same passage / query / none"""
        U = len(units)
        pid = torch.tensor([u["pid"] for u in units]); sid = torch.tensor([u["sid"] for u in units])
        same = pid.unsqueeze(0) == pid.unsqueeze(1)
        adj = same & ((sid.unsqueeze(0) - sid.unsqueeze(1)).abs() == 1)
        R = torch.full((U, U), R_NONE, dtype=torch.long)
        R[same] = R_PASS; R[adj] = R_ADJ
        R[torch.arange(U), torch.arange(U)] = R_SELF
        R[0, :] = R_QU; R[:, 0] = R_QU; R[0, 0] = R_QQ
        return R

    @staticmethod
    def relations(ex, units):
        U = len(units)
        R = torch.full((U, U), R_NONE, dtype=torch.long)
        R[0, :] = R_QU; R[:, 0] = R_QU; R[0, 0] = R_QQ
        idx = {(u["pid"], u["sid"]): i for i, u in enumerate(units)}
        ents, titles = {}, {}
        for i, u in enumerate(units[1:], 1):
            p = ex["passages"][u["pid"]]
            text = p["title"] if u["kind"] == "title" else p["sents"][u["sid"]]
            ents[i] = entities(text)
            titles[u["pid"]] = p["title"].lower()
        for i, ui in enumerate(units[1:], 1):
            for j, uj in enumerate(units[1:], 1):
                if i == j:
                    R[i, j] = R_SELF
                elif ui["pid"] == uj["pid"]:
                    R[i, j] = R_ADJ if abs(ui["sid"] - uj["sid"]) == 1 else R_PASS
                else:
                    tj = titles[uj["pid"]]
                    if (ents[i] & ents[j]) or (len(tj) > 3 and tj in ents[i]):
                        R[i, j] = R_ENT
        for pi, p in enumerate(ex["passages"]):
            for a, b in dependency_edges(p["sents"]):
                if (pi, a) in idx and (pi, b) in idx:
                    R[idx[(pi, a)], idx[(pi, b)]] = R_DEP
                    R[idx[(pi, b)], idx[(pi, a)]] = R_DEP
        return R


def dependency_list(ex, units, anchors_only=False):
    """Hard fidelity constraints keep(u) => keep(v) over unit indices."""
    idx = {(u["pid"], u["sid"]): i for i, u in enumerate(units)}
    deps = []
    for i, u in enumerate(units):
        if u["kind"] == "sent":
            deps.append((i, idx[(u["pid"], -1)]))  # entity anchoring: sentence -> its title
    if anchors_only:
        return deps
    for pi, p in enumerate(ex["passages"]):
        for a, b in dependency_edges(p["sents"]):
            if (pi, a) in idx and (pi, b) in idx:
                deps.append((idx[(pi, a)], idx[(pi, b)]))
    return deps


def _mlp(d):
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))


class CAET(nn.Module):
    def __init__(self, backbone, n_global=4, grad_ckpt=True):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone, add_pooling_layer=False, attn_implementation="sdpa")
        cfg = self.enc.config
        self.L, self.K, self.H, self.d = cfg.num_hidden_layers, n_global, cfg.num_attention_heads, cfg.hidden_size
        self.rel_bias = nn.Parameter(torch.zeros(N_REL, self.H))          # beta_{h,r}
        self.sel_emb = nn.Embedding(3, self.d)                              # query / kept / dropped
        nn.init.zeros_(self.sel_emb.weight)
        self.keep_head, self.omit_head, self.suff_head = _mlp(self.d), _mlp(self.d), _mlp(self.d)
        self.grad_ckpt = grad_ckpt
        # neural CRF: learned non-negative pairwise couplings c_uv (submodular => exact min-cut MAP)
        self.pair_rank = 128
        self.pair_q = nn.Linear(self.d, self.pair_rank); self.pair_k = nn.Linear(self.d, self.pair_rank)
        self.pair_rel = nn.Parameter(torch.zeros(N_REL))
        self.pair_bias = nn.Parameter(torch.tensor(-4.0))
        # contextual-dependency head: D_uv = P(unit u cannot be read correctly without unit v)  (directed)
        self.dep_q = nn.Linear(self.d, self.pair_rank); self.dep_k = nn.Linear(self.d, self.pair_rank)
        self.dep_rel = nn.Parameter(torch.zeros(N_REL))
        self.dep_bias = nn.Parameter(torch.tensor(-2.0))

    # ---- helpers ----
    def _run(self, layers, h, mask):
        for layer in layers:
            if self.grad_ckpt and self.training:
                h = checkpoint(lambda x, m, l=layer: l(x, attention_mask=m)[0], h, mask, use_reentrant=False)
            else:
                h = layer(h, attention_mask=mask)[0]
        return h

    @staticmethod
    def _pool(h, uid, U):
        s = torch.zeros(U, h.size(-1), device=h.device, dtype=h.dtype).index_add_(0, uid, h)
        c = torch.zeros(U, device=h.device, dtype=h.dtype).index_add_(0, uid, torch.ones_like(uid, dtype=h.dtype))
        return s / c.clamp(min=1).unsqueeze(-1)

    def _neg(self, dtype):
        return torch.finfo(dtype).min / 2

    # ---- passes ----
    def read(self, f):
        """Local reading + relational reasoning. Returns keep logits per unit and cached states."""
        dev = self.rel_bias.device
        ids, uid, pid, R = f["input_ids"].to(dev), f["uid"].to(dev), f["pid"].to(dev), f["R"].to(dev)
        U = f["n_units"]
        h = self.enc.embeddings(input_ids=ids.unsqueeze(0))
        dt = h.dtype if not torch.is_autocast_enabled() else torch.bfloat16
        neg = self._neg(dt)
        # block-diagonal local mask: token i sees j iff same passage or j is query
        local = (pid.unsqueeze(1) == pid.unsqueeze(0)) | (pid.unsqueeze(0) == -1)
        local_mask = torch.zeros(local.shape, device=dev, dtype=dt).masked_fill(~local, neg)[None, None]
        h_local = self._run(self.enc.encoder.layer[: self.L - self.K], h, local_mask)
        rel_tok = R[uid][:, uid]                                         # T x T relation ids
        bias = self.rel_bias.to(dt)[rel_tok].permute(2, 0, 1).unsqueeze(0)  # 1 x H x T x T
        h_glob = self._run(self.enc.encoder.layer[self.L - self.K:], h_local, bias)
        hu = self._pool(h_glob[0], uid, U)
        keep = self.keep_head(hu).squeeze(-1)
        return keep, dict(h_local=h_local, rel_tok=rel_tok, uid=uid, U=U, dt=dt, hu=hu, R=R)

    def couplings(self, cache):
        """symmetric non-negative coupling matrix C (U x U), zero diagonal"""
        hu = cache["hu"].float()
        qk = self.pair_q.float()(hu) @ self.pair_k.float()(hu).T / self.pair_rank ** 0.5
        logits = 0.5 * (qk + qk.T) + self.pair_rel.float()[cache["R"]] + self.pair_bias.float()
        C = F.softplus(logits)
        C = C * (1 - torch.eye(C.size(0), device=C.device))
        C[0, :] = 0; C[:, 0] = 0
        return C

    def dependency_logits(self, cache):
        """directed dependency logits (U x U): row u = the unit whose reading may need column v"""
        hu = cache["hu"].float()
        return (self.dep_q.float()(hu) @ self.dep_k.float()(hu).T / self.pair_rank ** 0.5
                + self.dep_rel.float()[cache["R"]] + self.dep_bias.float())

    def dependencies(self, cache):
        D = torch.sigmoid(self.dependency_logits(cache))
        D = D * (1 - torch.eye(D.size(0), device=D.device))
        D[0, :] = 0; D[:, 0] = 0
        return D

    def verify(self, cache, kept):
        """Re-read evidence under the selection mask. kept: bool tensor [U] (unit 0 = query, always visible)."""
        uid, U, dt = cache["uid"], cache["U"], cache["dt"]
        kept = kept.clone(); kept[0] = True
        state = torch.where(kept, torch.ones_like(uid[:U]), 2 * torch.ones_like(uid[:U]))
        state[0] = 0
        tok_state = state[uid]
        h = cache["h_local"] + self.sel_emb(tok_state).to(cache["h_local"].dtype).unsqueeze(0)
        vis = kept[uid]                                                    # visible tokens
        allow = vis.unsqueeze(0).expand(len(uid), -1) | (uid.unsqueeze(1) == uid.unsqueeze(0))
        neg = self._neg(dt)
        bias = self.rel_bias.to(dt)[cache["rel_tok"]].permute(2, 0, 1)
        bias = bias.masked_fill(~allow.unsqueeze(0), neg).unsqueeze(0)
        h2 = self._run(self.enc.encoder.layer[self.L - self.K:], h, bias)
        hu = self._pool(h2[0], uid, U)
        omit = self.omit_head(hu).squeeze(-1)
        suff = self.suff_head(hu[0:1]).squeeze()
        return omit, suff
