"""
The full librarian on a REAL LLM's features.

Ports the end-to-end memory loop — surprise-spawn + evidence-based consolidation +
frozen semantic id + a distance/entropy defer gate — onto frozen `gpt2` (125M)
embeddings, on a DBpedia class stream. The librarian is a memory-management policy
over prototypes, so it runs directly on the embeddings (no LoRA / no relaxation);
the only change from the toy is that the "position" of a concept is a real
sentence embedding instead of a random projection.

Mixed stream: clean classes recurring with label noise, a wave of NOVEL classes
introduced halfway, and (for the defer metric) ambiguous queries = midpoints
between two consolidated ids. Compared to `naive` = one prototype per observation,
no consolidation, no defer.

Metrics: #ids vs #real classes (noise rejection), clean + novel accuracy
(nearest-id classification), ambiguous-query defer rate.

Downloads once, cached: gpt2 (~0.5GB), DBpedia (~150MB slice). MPS/CPU, minutes.

# ponytail: feature-space prototypes (Euclidean on unit-norm embeddings) — the LLM
# only provides the address; the loop is the same one the toy validated.
"""

import argparse
import json
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

MODELS = {"gpt2": ("gpt2", 768), "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384)}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_CLASSES = 14
PER_CLASS_TR, PER_CLASS_TE = 40, 20
MAXLEN, BATCH = 48, 16


@torch.no_grad()
def embed(model, tok, texts):
    out = []
    for i in range(0, len(texts), BATCH):
        e = tok(texts[i:i + BATCH], return_tensors="pt", padding="max_length",
                truncation=True, max_length=MAXLEN).to(DEVICE)
        h = model(**e).last_hidden_state
        mm = e["attention_mask"].unsqueeze(-1).float()
        out.append(F.normalize((h * mm).sum(1) / mm.sum(1), dim=-1).cpu())   # mean-pool
    return torch.cat(out)


def load_feats(model_key, seed):
    name, _ = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    m = AutoModel.from_pretrained(name).to(DEVICE).eval()
    g = torch.Generator().manual_seed(seed)
    out = {}
    for split, pc in (("train", PER_CLASS_TR), ("test", PER_CLASS_TE)):
        ds = load_dataset("fancyzhx/dbpedia_14", split=split)
        lab = torch.tensor(ds["label"])
        texts, ys = [], []
        for c in range(N_CLASSES):
            idx = (lab == c).nonzero().flatten()
            idx = idx[torch.randperm(len(idx), generator=g)[:pc]]
            texts += [ds[int(i)]["content"] for i in idx]
            ys += [c] * len(idx)
        out[split] = (embed(m, tok, texts), torch.tensor(ys))
    # Whiten with TRAIN statistics: decoder-LM (gpt2) embeddings are anisotropic —
    # all inputs pile into a tiny cone (within-class ≈ cross-class ≈ 0.05), which
    # collapses prototype memory. Mean-center + per-dim standardize + renormalize
    # lifts DBpedia NCM routing 0.52 → 0.91. (Harmless for the already-isotropic
    # sentence embedder.)
    mu, sd = out["train"][0].mean(0), out["train"][0].std(0) + 1e-6
    for split in out:
        X, y = out[split]
        out[split] = (F.normalize((X - mu) / sd, dim=-1), y)
    return out


def _entropy(centers, e, temp=0.05):
    d2 = ((centers - e) ** 2).sum(-1)
    p = torch.softmax(-d2 / temp, dim=-1)
    return float(-(p * (p + 1e-12).log()).sum())


class FeatureLibrarian:
    """Prototype memory over LLM embeddings. Surprise → provisional candidate that
    accrues evidence; consolidates to a frozen id after k hits with ≥c label
    agreement. A near+ambiguous input (high entropy over ids, close to a known one)
    is deferred — parked, not placed. `full=False` is the naive control: one
    prototype per observation, no evidence gate, no defer."""

    def __init__(self, full=True, k=3, c=0.5, radius=0.7, tau=0.5, temp=0.3):
        self.full, self.k, self.c, self.radius, self.tau, self.temp = full, k, c, radius, tau, temp
        self.ids, self.prov, self.n_defer = [], [], 0

    def _nearest(self, items, e):
        if not items:
            return None, math.inf
        d = [((it["center"] - e) ** 2).sum().item() for it in items]
        j = min(range(len(d)), key=lambda i: d[i])
        return j, d[j] ** 0.5

    def is_ambiguous(self, e):
        if len(self.ids) < 2:
            return False
        C = torch.stack([it["center"] for it in self.ids])
        _, dist = self._nearest(self.ids, e)
        return dist < self.radius and _entropy(C, e, self.temp) > self.tau

    def observe(self, e, y):
        if self.full and self.is_ambiguous(e):
            self.n_defer += 1
            return                                        # "I don't know" → park
        if not self.full:                                 # naive: a prototype per obs
            self.ids.append({"center": e.clone(), "label": y})
            return
        j, dist = self._nearest(self.ids, e)              # consolidated prediction
        if j is not None and self.ids[j]["label"] == y and dist < self.radius:
            return                                        # already known → reinforce (no-op)
        if j is not None and dist < self.radius:
            return                                        # near a known id → ignore labile obs
        pj, _ = self._nearest(self.prov, e)               # accrue provisional evidence
        if pj is not None and ((self.prov[pj]["center"] - e) ** 2).sum().sqrt().item() < self.radius:
            p = self.prov[pj]
            p["hits"] += 1
            p["counts"][y] = p["counts"].get(y, 0) + 1
            p["center"] = p["center"] + 0.5 * (e - p["center"])
        else:
            self.prov.append({"center": e.clone(), "counts": {y: 1}, "hits": 1})
            pj = len(self.prov) - 1
        p = self.prov[pj]
        if p["hits"] >= self.k and max(p["counts"].values()) / p["hits"] >= self.c:
            self.ids.append({"center": p["center"], "label": max(p["counts"], key=p["counts"].get)})
            self.prov.pop(pj)

    def predict(self, e):
        j, _ = self._nearest(self.ids, e)
        return self.ids[j]["label"] if j is not None else -1


def run(model_key, kind, seed, epochs=6, noise=0.15):
    data = load_feats(model_key, seed)
    Xtr, ytr = data["train"]
    Xte, yte = data["test"]
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 3)
    lib = FeatureLibrarian(full=(kind == "librarian"))
    by_class = [(Xtr[ytr == c], c) for c in range(N_CLASSES)]
    novel = list(range(10, N_CLASSES))                    # classes 10-13 arrive halfway
    for ep in range(epochs):
        active = range(N_CLASSES) if ep >= epochs // 2 else range(10)
        order = []
        for c in active:
            for e in by_class[c][0]:
                order.append((e, c))
        for idx in torch.randperm(len(order), generator=g):
            e, y = order[idx]
            if torch.rand(1, generator=g).item() < noise:
                y = int(torch.randint(0, N_CLASSES, (1,), generator=g).item())
            lib.observe(e, y)
    clean = (torch.tensor([lib.predict(e) for e in Xte]) == yte).float().mean().item()
    nmask = torch.tensor([int(c) in novel for c in yte])
    novel_acc = (torch.tensor([lib.predict(Xte[i]) for i in range(len(Xte)) if nmask[i]])
                 == yte[nmask]).float().mean().item()
    amb_defer = 0.0
    if kind == "librarian" and len(lib.ids) >= 2:
        C = torch.stack([it["center"] for it in lib.ids])
        probes = []
        for _ in range(60):
            i, j = torch.randint(0, len(lib.ids), (2,), generator=g).tolist()
            if i != j:
                probes.append((lib.ids[i]["center"] + lib.ids[j]["center"]) / 2)
        amb_defer = sum(_entropy(C, p, lib.temp) > lib.tau for p in probes) / max(len(probes), 1)
    return {"clean_acc": clean, "novel_acc": novel_acc, "n_ids": len(lib.ids),
            "amb_defer": amb_defer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="gpt2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    out = {"config": {"model": args.model, "seeds": seeds, "n_classes": N_CLASSES}, "summary": {}}
    for kind in ("librarian", "naive"):
        res = [run(args.model, kind, s) for s in seeds]
        out["summary"][kind] = {k: sum(r[k] for r in res) / len(res)
                                for k in ("clean_acc", "novel_acc", "n_ids", "amb_defer")}
        s = out["summary"][kind]
        print(f"{kind:10s} clean {s['clean_acc']:.2f}  novel {s['novel_acc']:.2f}  "
              f"ids {s['n_ids']:.0f}  amb_defer {s['amb_defer']:.2f}")
    path = args.out or f"results/llm_librarian_{args.model}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
