"""
The librarian on CODE models — local runnable version of the Colab notebook.

Continual stream over programming languages (CodeSearchNet, streamed to keep
downloads light), frozen code-model embeddings (mean-pool + whitening), the
librarian memory loop vs a naive one-prototype-per-observation baseline. Compares
an OLDER code model vs a newer code LLM on the identical task — the "diff".

Runs on MPS/CPU with small models. Downloads once, cached.
"""

import argparse
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from nous.train_llm_librarian import FeatureLibrarian, _entropy

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LANGS = ["python", "java", "go", "javascript", "php"]     # last = novel (introduced late)
PER, TEST = 100, 40
MODELS = {"codebert-2020": "microsoft/codebert-base",
          "qwen2.5-coder-0.5b-2024": "Qwen/Qwen2.5-Coder-0.5B",
          "qwen2.5-coder-1.5b-2024": "Qwen/Qwen2.5-Coder-1.5B"}


def load_code(seed=0):
    torch.manual_seed(seed)
    tr, te, ytr, yte = [], [], [], []
    for li, lang in enumerate(LANGS):
        ds = load_dataset("code_search_net", lang, split="train", streaming=True)
        docs = []
        for row in ds:
            docs.append(row.get("func_code_string") or row.get("whole_func_string") or "")
            if len(docs) >= PER + TEST:
                break
        tr += docs[:PER]; ytr += [li] * PER
        te += docs[PER:PER + TEST]; yte += [li] * (len(docs) - PER)
    return tr, te, torch.tensor(ytr), torch.tensor(yte)


@torch.no_grad()
def embed(model, tok, texts, maxlen=128, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                truncation=True, max_length=maxlen).to(DEVICE)
        h = model(**e).last_hidden_state
        mm = e["attention_mask"].unsqueeze(-1).float()
        out.append(F.normalize((h * mm).sum(1) / mm.sum(1), dim=-1).cpu())
    return torch.cat(out)


def features(name, tr, te):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModel.from_pretrained(name, trust_remote_code=True).to(DEVICE).eval()
    Xtr, Xte = embed(m, tok, tr), embed(m, tok, te)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6                # whiten (decoder-LM anisotropy fix)
    del m
    return F.normalize((Xtr - mu) / sd, dim=-1), F.normalize((Xte - mu) / sd, dim=-1)


def run_model(name, tr, te, ytr, yte, seed=0, epochs=6, noise=0.15):
    Xtr, Xte = features(name, tr, te)
    mu = torch.stack([Xtr[ytr == c].mean(0) for c in range(len(LANGS))])
    routing = (torch.cdist(Xte, mu).argmin(1) == yte).float().mean().item()
    g = torch.Generator().manual_seed(seed + 3)
    res = {"routing": routing}
    for kind in ("librarian", "naive"):
        lib = FeatureLibrarian(full=(kind == "librarian"))
        by = [Xtr[ytr == c] for c in range(len(LANGS))]
        for ep in range(epochs):
            active = range(len(LANGS)) if ep >= epochs // 2 else range(len(LANGS) - 1)
            order = [(e, c) for c in active for e in by[c]]
            for k in torch.randperm(len(order), generator=g):
                e, y = order[k]
                if torch.rand(1, generator=g).item() < noise:
                    y = int(torch.randint(0, len(LANGS), (1,), generator=g).item())
                lib.observe(e, y)
        clean = (torch.tensor([lib.predict(e) for e in Xte]) == yte).float().mean().item()
        nmask = yte == (len(LANGS) - 1)
        novel = (torch.tensor([lib.predict(Xte[i]) for i in range(len(Xte)) if nmask[i]])
                 == yte[nmask]).float().mean().item()
        amb = 0.0
        if kind == "librarian" and len(lib.ids) >= 2:
            C = torch.stack([it["center"] for it in lib.ids])
            pr = [(lib.ids[i]["center"] + lib.ids[j]["center"]) / 2
                  for i, j in (torch.randint(0, len(lib.ids), (2,), generator=g).tolist()
                               for _ in range(60)) if i != j]
            amb = sum(_entropy(C, p, lib.temp) > lib.tau for p in pr) / max(len(pr), 1)
        res[kind] = dict(clean=clean, novel=novel, ids=len(lib.ids), defer=amb)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tr, te, ytr, yte = load_code(args.seed)
    print(f"langs {LANGS} | train {len(tr)} test {len(te)}")
    for label, name in MODELS.items():
        t = time.time()
        r = run_model(name, tr, te, ytr, yte, args.seed)
        L, N = r["librarian"], r["naive"]
        print(f"[{time.time()-t:.0f}s] {label:26s} routing {r['routing']:.2f} | "
              f"librarian clean {L['clean']:.2f} novel {L['novel']:.2f} ids {L['ids']:4d} "
              f"defer {L['defer']:.2f} | naive clean {N['clean']:.2f} ids {N['ids']:4d}")


if __name__ == "__main__":
    main()
