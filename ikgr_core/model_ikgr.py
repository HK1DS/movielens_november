import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Robust imports for different RecBole versions
try:
    from recbole.model.general_recommender import GeneralRecommender
except Exception:
    try:
        from recbole.model.abstract_recommender import GeneralRecommender  # very old fallback
    except Exception as e:
        raise ImportError(
            "Cannot import GeneralRecommender from RecBole. "
            "Please upgrade RecBole (pip install 'recbole>=1.2.0')."
        ) from e

try:
    from recbole.utils import InputType
except Exception:
    # Fallback enum stub if very old RecBole; PAIRWISE is common in BPR-like training
    class InputType:
        PAIRWISE = type("X", (), {"value": "pairwise"})
        POINTWISE = type("X", (), {"value": "pointwise"})

def _cfg(config, key: str, default):
    """Safe access for RecBole Config: supports `key in config` + `config[key]`."""
    return config[key] if key in config else default


class IKGR(GeneralRecommender):
    """
    IKGR (intent knowledge-graph recommender), redesigned to be LEARNABLE and to
    actually use the intent KG:

      - Base learnable user/item embeddings (like MF/BPR).
      - Shared learnable INTENT NODES initialized from the LLM/mpnet intent
        embeddings (run/kg_pack.pt) via a learnable projection.
      - One vectorized KGCN-style propagation layer: each user/item embedding is
        enriched by the (row-normalized) mean of its connected intent-node
        embeddings -> u' = e_u + a * (A_u_hat @ E_intent), i' = e_i + a * ...
      - Score = <u', i'> (inner product), trained with BPR. No frozen heuristic.

    Set `use_kg=False` to disable propagation (-> plain MF/BPR), which gives the
    KG-on vs KG-off ablation with everything else identical.

    Why the rewrite: the previous version scored y_ui (frozen max-cosine over
    intent pairs, range ~0.5-1.0) + 0.1*cos(emb). The frozen term dominated and
    saturated (a 0.9 penalty cliff tied hundreds of items at the top per user),
    so the learnable signal was suppressed and it ranked below Pop.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embed_dim = int(_cfg(config, "embedding_size", 64))
        self.dropout = float(_cfg(config, "dropout_prob", 0.1))
        self.reg_weight = float(_cfg(config, "reg_weight", 1e-6))
        self.use_kg = bool(_cfg(config, "use_kg", True))
        self.kg_pack_path = _cfg(config, "kg_pack_path", "run/kg_pack.pt")
        self.kg_layers = int(_cfg(config, "kg_layers", 1))
        self.intent_learnable = bool(_cfg(config, "intent_learnable", True))
        self.kg_cap = int(_cfg(config, "kg_cap", 64))
        # Heterogeneous metadata KG (brand/category/attribute), LLM-free.
        self.use_meta_kg = bool(_cfg(config, "use_meta_kg", False))
        self.meta_kg_path = _cfg(config, "meta_kg_path", "run/meta_kg_pack.pt")
        self._meta_ready = False
        # DynLLM-style dynamic profile: recency-weighted aggregation of the
        # user's recent (train-only) item embeddings. Injected via set_recency().
        self.use_dynamic = bool(_cfg(config, "use_dynamic", False))
        self.recency_topn = int(_cfg(config, "recency_topn", 50))
        self.recency_tau_days = float(_cfg(config, "recency_tau_days", 180.0))
        self._recency_ready = False
        if self.use_dynamic:
            self.dynamic_alpha = nn.Parameter(torch.tensor(1.0))
        # Step 3: multi-facet attention fusion (DynLLM multi_profile_target).
        # Replaces the fixed scalar-gate sum with query=base, key/value=facets MHA.
        self.profile_attn = bool(_cfg(config, "profile_attn", False))
        if self.profile_attn:
            nh = int(_cfg(config, "profile_attn_heads", 4))
            self.user_attn = nn.MultiheadAttention(self.embed_dim, nh, dropout=self.dropout, batch_first=True)
            self.item_attn = nn.MultiheadAttention(self.embed_dim, nh, dropout=self.dropout, batch_first=True)
        # CORONA (stage 3) weighted-sum late-fusion = the Full(IKGR+DynLLM+CORONA)
        # model. Instead of one inner product over summed facet embeddings (the
        # default _fuse path, which implicitly mixes every cross term at weight 1),
        # score each CHANNEL separately and combine with learnable weights:
        #   Final = gamma * <e_u, e_i>            (CF / item-similarity channel)
        #         + alpha * <kg_u, kg_i>          (intent + meta-KG channel)
        #         + beta  * <dyn_u, e_i>          (DynLLM recency channel)
        # This is the diagram's 3-3 weighted-sum ranking. The honest question
        # (CORONA_INTEGRATION.md sec.6) is whether explicit separation + learned
        # weights beats the implicit summed-embedding score.
        self.use_corona = bool(_cfg(config, "use_corona", False))
        if self.use_corona:
            self.corona_alpha = nn.Parameter(torch.tensor(1.0))   # intent/KG channel
            self.corona_beta = nn.Parameter(torch.tensor(1.0))    # dynamic channel
            self.corona_gamma = nn.Parameter(torch.tensor(1.0))   # CF/base channel
        # LightGCN backbone (lgcn_layers > 0): the CF base embeddings are propagated
        # over the TRAIN user-item graph exactly as RecBole's LightGCN does
        # (symmetric-normalized adjacency, layer-mean), and the KG/metadata/recency
        # facets are added on top of the propagated representations instead of the
        # raw MF embeddings. lgcn_layers=0 (default) leaves every existing spec unchanged.
        self.lgcn_layers = int(_cfg(config, "lgcn_layers", 0))
        self.sparse_format = str(_cfg(config, "sparse_format", "coo") or "coo").lower()   # coo | csr
        if self.lgcn_layers > 0 and (self.use_corona or self.profile_attn
                                     or (bool(_cfg(config, "use_kg", True)) and int(_cfg(config, "kg_layers", 1)) >= 2)):
            raise ValueError("lgcn_layers > 0 is only supported with the default additive fusion "
                             "(no use_corona / profile_attn / kg_layers>=2)")

        n_users, n_items = self.n_users, self.n_items
        self.user_embedding = nn.Embedding(n_users, self.embed_dim)
        self.item_embedding = nn.Embedding(n_items, self.embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.dropout_layer = nn.Dropout(self.dropout)

        self._kg_ready = False
        if self.use_kg:
            self._build_kg(dataset)
        if self.use_meta_kg:
            self._build_meta_kg(dataset)
        self._adj = None
        if self.lgcn_layers > 0:
            self._build_lgcn_adj(dataset)
        # Full-sort eval cache: all-item representations are expensive to recompute
        # (KG/meta node gather over every item) and were being rebuilt for EVERY
        # test-user batch. Cache them once per eval pass (set via eval_cache_items).
        self._item_cache_on = False
        self._cache_i_all = None
        self._cache_icf = None
        self._cache_ikg = None
        self._cache_u_all = None
        self._cache_lg = None

    # ---- KG construction (learnable intent nodes + normalized user/item<->intent adjacency)
    def _build_kg(self, dataset):
        pack = torch.load(self.kg_pack_path, map_location="cpu")
        raw = pack["intent_emb"].float()             # [n_intents, in_dim] (mpnet)
        self.n_intents = int(pack["n_intents"])
        in_dim = raw.shape[1]
        d = self.embed_dim

        # Intent NODE embeddings, initialized from a fixed random projection of
        # the LLM/mpnet intent vectors (preserves semantic geometry, JL lemma),
        # then learned jointly (genuine shared KG nodes).
        if in_dim == d:
            init = raw.clone()
        else:
            g = torch.Generator().manual_seed(12345)
            P = torch.randn(in_dim, d, generator=g) / (d ** 0.5)
            init = raw @ P                           # [n_intents, d]
        self.intent_embedding = nn.Embedding(self.n_intents, d)
        self.intent_embedding.weight.data.copy_(init)
        if not self.intent_learnable:
            self.intent_embedding.weight.requires_grad_(False)
        self.intent_alpha = nn.Parameter(torch.tensor(1.0))

        u_tok2id = dataset.field2token_id[self.USER_ID]
        i_tok2id = dataset.field2token_id[self.ITEM_ID]

        def edges(token_intents, tok2id):
            rs, cs = [], []
            for tok, ids in token_intents.items():
                rid = tok2id.get(str(tok))
                if rid is None:
                    continue
                ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
                for j in ids:
                    rs.append(rid); cs.append(int(j))
            return rs, cs

        ur, uc = edges(pack["user_intents"], u_tok2id)   # user<->intent edges
        ir, ic = edges(pack["item_intents"], i_tok2id)   # item<->intent edges

        def norm_coo(rows, cols, n_rows):
            """Row-normalized COO: value = 1/deg(row)."""
            rt = torch.tensor(rows, dtype=torch.long)
            ct = torch.tensor(cols, dtype=torch.long)
            deg = torch.zeros(n_rows)
            deg.index_add_(0, rt, torch.ones(len(rows)))
            val = 1.0 / deg[rt].clamp(min=1.0)
            return torch.stack([rt, ct]), val

        # Mu : user-normalized  user->intent  (user gathers mean of its intents)
        # MuT: intent-normalized intent->user (intent gathers mean of its users)
        mats = {
            "Mu":  norm_coo(ur, uc, self.n_users),
            "Mi":  norm_coo(ir, ic, self.n_items),
            "MuT": norm_coo(uc, ur, self.n_intents),
            "MiT": norm_coo(ic, ir, self.n_intents),
        }
        for name, (idx, val) in mats.items():
            self.register_buffer(f"{name}_idx", idx)
            self.register_buffer(f"{name}_val", val)

        # Padded intent-id tensors for FAST mini-batch L1 aggregation (gather the
        # batch's intent-node embeddings directly, avoiding a full-graph sparse
        # mm every step). Equivalent to Mu/Mi row-mean but only for batch rows.
        def padded(rows, cols, n_rows, cap):
            from collections import defaultdict
            d = defaultdict(list)
            for r, c in zip(rows, cols):
                if len(d[r]) < cap:
                    d[r].append(c)
            ids = torch.zeros(n_rows, cap, dtype=torch.long)
            mask = torch.zeros(n_rows, cap, dtype=torch.bool)
            for r, lst in d.items():
                k = len(lst)
                ids[r, :k] = torch.tensor(lst, dtype=torch.long)
                mask[r, :k] = True
            return ids, mask
        ui, um = padded(ur, uc, self.n_users, self.kg_cap)
        ii, im = padded(ir, ic, self.n_items, self.kg_cap)
        self.register_buffer("user_intent_ids", ui)
        self.register_buffer("user_intent_mask", um)
        self.register_buffer("item_intent_ids", ii)
        self.register_buffer("item_intent_mask", im)
        self._kg_ready = True

    def _sp(self, name, shape):
        return torch.sparse_coo_tensor(getattr(self, f"{name}_idx"),
                                       getattr(self, f"{name}_val"), shape)

    def _agg(self, emb, ids_buf, mask_buf, rows):
        ids = ids_buf[rows]                                  # [B, cap]
        m = mask_buf[rows].unsqueeze(-1).float()             # [B, cap, 1]
        e = emb(ids) * m                                     # [B, cap, d]
        return e.sum(1) / m.sum(1).clamp(min=1.0)            # [B, d]

    def _build_meta_kg(self, dataset):
        """Heterogeneous item-side KG from the adapter's static item metadata
        (build_meta_kg.py, pack format 2: one node type per relation).
        Per relation: learnable node embeddings + a learnable gate + padded
        item->node gather (LLM-free). Relations with no edge on the current
        item vocabulary are skipped."""
        pack = torch.load(self.meta_kg_path, map_location="cpu")
        if "relations" not in pack:
            raise ValueError(f"{self.meta_kg_path}: expected meta-KG pack format 2 "
                             "({'relations': {...}}); rebuild it with build_meta_kg.py")
        i_tok2id = dataset.field2token_id[self.ITEM_ID]
        d, cap = self.embed_dim, self.kg_cap
        self.meta_embeddings = nn.ModuleDict()
        self.meta_alphas = nn.ParameterDict()
        self.meta_rels = []                      # module keys, in deterministic (sorted) order
        for rel in sorted(pack["relations"]):
            r = pack["relations"][rel]
            key = re.sub(r"[^0-9A-Za-z_]", "_", str(rel)) or "rel"
            if key in self.meta_embeddings:
                raise ValueError(f"meta relations {rel!r} and another relation map to the same key {key!r}")
            ids = torch.zeros(self.n_items, cap, dtype=torch.long)
            mask = torch.zeros(self.n_items, cap, dtype=torch.bool)
            present = False
            for tok, node_ids in r["item_nodes"].items():
                rid = i_tok2id.get(str(tok))
                if rid is None:
                    continue
                lst = (node_ids.tolist() if torch.is_tensor(node_ids) else list(node_ids))[:cap]
                if not lst:
                    continue
                present = True
                ids[rid, :len(lst)] = torch.tensor(lst, dtype=torch.long)
                mask[rid, :len(lst)] = True
            if not present:
                continue
            emb = nn.Embedding(max(int(r["n_nodes"]), 1), d); nn.init.xavier_uniform_(emb.weight)
            self.meta_embeddings[key] = emb
            self.meta_alphas[key] = nn.Parameter(torch.tensor(1.0))
            self.register_buffer(f"meta_ids_{key}", ids)
            self.register_buffer(f"meta_mask_{key}", mask)
            self.meta_rels.append(key)
        self._meta_ready = True

    def _meta_parts(self, iids):
        """[(gate, aggregated node embedding [B, d])] for every metadata relation."""
        if not (self.use_meta_kg and self._meta_ready):
            return []
        return [(self.meta_alphas[k],
                 self._agg(self.meta_embeddings[k], getattr(self, f"meta_ids_{k}"),
                           getattr(self, f"meta_mask_{k}"), iids))
                for k in self.meta_rels]

    def _build_lgcn_adj(self, dataset):
        """Symmetric-normalized bipartite adjacency of the TRAIN interactions,
        D^-1/2 [[0, R], [R^T, 0]] D^-1/2 (same construction as RecBole LightGCN).
        `dataset` is the train split's dataset, so no valid/test edge enters."""
        import numpy as np
        R = dataset.inter_matrix(form="coo").astype(np.float32).tocsr()
        R.data[:] = 1.0                       # binary, duplicates collapsed
        R = R.tocoo()
        nu, ni = self.n_users, self.n_items
        rows = np.concatenate([R.row, R.col + nu])
        cols = np.concatenate([R.col + nu, R.row])
        deg = np.bincount(rows, minlength=nu + ni).astype(np.float32) + 1e-7
        dinv = np.power(deg, -0.5)
        vals = dinv[rows] * dinv[cols]
        self._adj_idx = torch.from_numpy(np.stack([rows, cols]).astype(np.int64))
        self._adj_val = torch.from_numpy(vals.astype(np.float32))
        self._adj_shape = (nu + ni, nu + ni)

    def _lgcn_adj(self):
        dev = self.user_embedding.weight.device
        if self._adj is None or self._adj.device != dev:
            adj = torch.sparse_coo_tensor(self._adj_idx, self._adj_val, self._adj_shape).coalesce().to(dev)
            self._adj = adj.to_sparse_csr() if self.sparse_format == "csr" else adj
        return self._adj

    def _lgcn_all(self):
        """(U, I) = layer-mean of LightGCN propagation of the base CF embeddings."""
        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        A = self._lgcn_adj()
        acc, cur = E, E
        for _ in range(self.lgcn_layers):
            cur = torch.sparse.mm(A, cur)
            acc = acc + cur
        out = acc / (self.lgcn_layers + 1)
        return torch.split(out, [self.n_users, self.n_items], dim=0)

    def set_recency(self, item_ids, weights):
        """Inject per-user recency profile (train-only). item_ids/weights:
        [n_users, recency_topn] (weights are recency-normalized, 0 for padding)."""
        self.register_buffer("recency_item_ids", item_ids.long())
        self.register_buffer("recency_weights", weights.float())
        self._recency_ready = True

    def _fuse(self, base, gated, raw, attn):
        """Combine base with facets. Default: base + sum(gated). If profile_attn:
        base + MHA(query=base, key/value=stacked raw facets)."""
        if not raw:
            return base
        if self.profile_attn:
            F_ = torch.stack(raw, dim=1)                      # [B, n_facets, d]
            out, _ = attn(base.unsqueeze(1), F_, F_)          # [B, 1, d]
            return base + out.squeeze(1)
        e = base
        for g in gated:
            e = e + g
        return e

    def _emb_users(self, uids, lg=None):
        """lg = (U, I) LightGCN-propagated tables when lgcn_layers > 0, else None."""
        base = self.user_embedding(uids) if lg is None else lg[0][uids]
        if self.use_kg and self._kg_ready and self.kg_layers >= 2:
            base = self._propagate()[0][uids]
        gated, raw = [], []
        if self.use_kg and self._kg_ready and self.kg_layers == 1:
            v = self._agg(self.intent_embedding, self.user_intent_ids, self.user_intent_mask, uids)
            gated.append(self.intent_alpha * v); raw.append(v)
        if self.use_dynamic and self._recency_ready:
            rid = self.recency_item_ids[uids]
            w = self.recency_weights[uids].unsqueeze(-1)
            item_rep = self.item_embedding(rid) if lg is None else lg[1][rid]
            dyn = (item_rep * w).sum(1)
            gated.append(self.dynamic_alpha * dyn); raw.append(dyn)
        return self._fuse(base, gated, raw, getattr(self, "user_attn", None))

    def _emb_items(self, iids, lg=None):
        base = self.item_embedding(iids) if lg is None else lg[1][iids]
        if self.use_kg and self._kg_ready and self.kg_layers >= 2:
            return self._propagate()[1][iids]
        gated, raw = [], []
        if self.use_kg and self._kg_ready:
            v = self._agg(self.intent_embedding, self.item_intent_ids, self.item_intent_mask, iids)
            gated.append(self.intent_alpha * v); raw.append(v)
        for alpha, v in self._meta_parts(iids):
            gated.append(alpha * v); raw.append(v)
        return self._fuse(base, gated, raw, getattr(self, "item_attn", None))

    # ---- CORONA (stage 3) per-channel reps + weighted-sum late-fusion ----
    def _corona_user_channels(self, uids):
        """Return (cf, kg, dyn) user channel reps (kg/dyn may be None)."""
        cf = self.dropout_layer(self.user_embedding(uids))
        kg = None
        if self.use_kg and self._kg_ready:
            kg = self._agg(self.intent_embedding, self.user_intent_ids, self.user_intent_mask, uids)
        dyn = None
        if self.use_dynamic and self._recency_ready:
            rid = self.recency_item_ids[uids]
            w = self.recency_weights[uids].unsqueeze(-1)
            dyn = (self.item_embedding(rid) * w).sum(1)
        return cf, kg, dyn

    def _corona_item_channels(self, iids):
        """Return (cf, kg) item channel reps. The dynamic channel scores against
        the item CF rep, so no separate item dyn rep is needed."""
        cf = self.dropout_layer(self.item_embedding(iids))
        parts = []
        if self.use_kg and self._kg_ready:
            parts.append(self._agg(self.intent_embedding, self.item_intent_ids, self.item_intent_mask, iids))
        parts.extend(v for _, v in self._meta_parts(iids))
        kg = None
        if parts:
            kg = parts[0]
            for p in parts[1:]:
                kg = kg + p
        return cf, kg

    def _corona_pair(self, uids, iids):
        ucf, ukg, udyn = self._corona_user_channels(uids)
        icf, ikg = self._corona_item_channels(iids)
        s = self.corona_gamma * (ucf * icf).sum(dim=-1)
        if ukg is not None and ikg is not None:
            s = s + self.corona_alpha * (ukg * ikg).sum(dim=-1)
        if udyn is not None:
            s = s + self.corona_beta * (udyn * icf).sum(dim=-1)
        return s

    def _corona_full(self, uids):
        ucf, ukg, udyn = self._corona_user_channels(uids)
        self._ensure_item_cache()
        icf, ikg = self._cache_icf, self._cache_ikg
        s = self.corona_gamma * torch.matmul(ucf, icf.t())
        if ukg is not None and ikg is not None:
            s = s + self.corona_alpha * torch.matmul(ukg, ikg.t())
        if udyn is not None:
            s = s + self.corona_beta * torch.matmul(udyn, icf.t())
        return s

    def _ensure_item_cache(self):
        """Lazily build the all-item representation cache during an eval pass.
        full_sort_predict is only called in eval mode (post-fit loop AND RecBole's
        in-training validation), so caching here avoids rebuilding every item's
        KG/meta-gathered embedding for every user batch. The cache is invalidated
        at the start of each training step (calculate_loss/forward), so it is
        always rebuilt with up-to-date weights for the next validation."""
        if self._item_cache_on:
            return
        with torch.no_grad():
            dev = self.item_embedding.weight.device
            all_items = torch.arange(self.n_items, device=dev)
            if self.use_corona:
                self._cache_icf, self._cache_ikg = self._corona_item_channels(all_items)
            elif self.use_kg and self._kg_ready and self.kg_layers >= 2:
                self._cache_u_all, self._cache_i_all = self._propagate()
            elif self.lgcn_layers > 0:
                self._cache_lg = self._lgcn_all()
                self._cache_i_all = self._emb_items(all_items, self._cache_lg)
            else:
                self._cache_i_all = self._emb_items(all_items)
        self._item_cache_on = True

    def eval_cache_items(self):
        """Force a fresh all-item cache (explicit entry point for eval harness)."""
        self._item_cache_on = False
        self._ensure_item_cache()

    def clear_eval_cache(self):
        self._item_cache_on = False
        self._cache_i_all = self._cache_icf = self._cache_ikg = None
        self._cache_lg = None

    def _propagate(self):
        """Return (u_all, i_all): KG-enriched user/item embeddings.

        L=1: users/items gather a mean of their connected intent nodes.
        L=2: intents additionally gather from their users+items, then propagate
             back -> user->intent->item collaborative signal flows across hops.
        """
        E_u = self.user_embedding.weight
        E_i = self.item_embedding.weight
        if not (self.use_kg and self._kg_ready):
            return E_u, E_i
        ni, nu, nit = self.n_intents, self.n_users, self.n_items
        Mu = self._sp("Mu", (nu, ni)); Mi = self._sp("Mi", (nit, ni))
        E_int = self.intent_embedding.weight

        u_acc = torch.sparse.mm(Mu, E_int)           # [n_users, d]
        i_acc = torch.sparse.mm(Mi, E_int)           # [n_items, d]
        if self.kg_layers >= 2:
            MuT = self._sp("MuT", (ni, nu)); MiT = self._sp("MiT", (ni, nit))
            int1 = torch.sparse.mm(MuT, E_u) + torch.sparse.mm(MiT, E_i)  # intents <- users+items
            u_acc = u_acc + torch.sparse.mm(Mu, int1)
            i_acc = i_acc + torch.sparse.mm(Mi, int1)

        u_all = E_u + self.intent_alpha * u_acc
        i_all = E_i + self.intent_alpha * i_acc
        return u_all, i_all

    def forward(self, user, item):
        self._item_cache_on = False
        if self.use_corona:
            return self._corona_pair(user, item)
        lg = self._lgcn_all() if self.lgcn_layers > 0 else None
        u = self.dropout_layer(self._emb_users(user, lg))
        i = self.dropout_layer(self._emb_items(item, lg))
        return (u * i).sum(dim=-1)

    def calculate_loss(self, interaction):
        self._item_cache_on = False
        user = interaction[self.USER_ID]
        pos = interaction[self.ITEM_ID]
        has_neg = hasattr(self, "NEG_ITEM_ID") and (self.NEG_ITEM_ID in interaction)

        if self.use_corona:
            pos_score = self._corona_pair(user, pos)
            if has_neg:
                neg = interaction[self.NEG_ITEM_ID]
                neg_score = self._corona_pair(user, neg)
                loss = -F.logsigmoid(pos_score - neg_score).mean()
                reg = (self.user_embedding(user).norm(2).pow(2)
                       + self.item_embedding(pos).norm(2).pow(2)
                       + self.item_embedding(neg).norm(2).pow(2))
            else:
                label = interaction[self._label_field()].float()
                loss = F.binary_cross_entropy_with_logits(pos_score, label)
                reg = (self.user_embedding(user).norm(2).pow(2)
                       + self.item_embedding(pos).norm(2).pow(2))
            return loss + self.reg_weight * reg / user.shape[0]

        lg = self._lgcn_all() if self.lgcn_layers > 0 else None   # one propagation per step
        u = self._emb_users(user, lg)
        pos_e = self._emb_items(pos, lg)
        pos_score = (u * pos_e).sum(dim=-1)

        if has_neg:
            neg = interaction[self.NEG_ITEM_ID]
            neg_e = self._emb_items(neg, lg)
            neg_score = (u * neg_e).sum(dim=-1)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            reg = (u.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2))
        else:
            label = interaction[self._label_field()].float()
            loss = F.binary_cross_entropy_with_logits(pos_score, label)
            reg = (u.norm(2).pow(2) + pos_e.norm(2).pow(2))

        return loss + self.reg_weight * reg / user.shape[0]

    def _label_field(self):
        if hasattr(self, "LABEL"):
            return self.LABEL
        if "LABEL_FIELD" in self.config:
            return self.config["LABEL_FIELD"]
        raise KeyError("Pointwise training requires LABEL_FIELD in config.")

    def predict(self, interaction):
        if self.use_corona:
            return self._corona_pair(interaction[self.USER_ID], interaction[self.ITEM_ID])
        lg = self._lgcn_all() if self.lgcn_layers > 0 else None
        u = self._emb_users(interaction[self.USER_ID], lg)
        i = self._emb_items(interaction[self.ITEM_ID], lg)
        return (u * i).sum(dim=-1)

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        if self.use_corona:
            return self._corona_full(users)
        self._ensure_item_cache()
        if self.use_kg and self._kg_ready and self.kg_layers >= 2:
            u_all = self._propagate()[0]
            return torch.matmul(u_all[users], self._cache_i_all.t())
        u = self._emb_users(users, self._cache_lg if self.lgcn_layers > 0 else None)   # [B, d]
        return torch.matmul(u, self._cache_i_all.t())

