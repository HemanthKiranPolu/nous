"""
NOUS-CLS on a pretrained transformer — the pretrained-model rung.

Same mechanism as the from-scratch transformer step, now on a REAL pretrained
backbone with REAL LoRA (peft): frozen `distilbert-base-uncased`, per-REGION
LoRA experts routed by geometry on the frozen [CLS] feature (no task id at test),
vs a single shared LoRA and full fine-tuning.

Task stream: 20 Newsgroups (sklearn, cached locally) → 5 tasks of 4 classes each,
presented in phases. After each phase, accuracy on all tasks so far.

Learners:
  per_region : one peft-LoRA adapter (q_lin,v_lin) + head per discovered region;
               only the routed region trains. Test routing is geometric.
  shared     : one LoRA adapter + one growing head across the whole stream.
  full_ft    : unfreeze the whole backbone + one growing head (upper-bound forget).

Downloads (once, then cached): distilbert-base-uncased (~270MB), 20NG (~14MB).
Runs on MPS/CPU; small subset keeps it to a few minutes.

# ponytail: real LoRA in attention (q_lin,v_lin) but a small data subset + few
# epochs + 3 seeds — enough to show the retention gap, not a leaderboard run.
# ponytail: experts still spawn at task boundaries; per-example surprise-spawn is
# the last remaining crutch (separate step). Test routing is already label-free.
"""

import argparse
import gc
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "distilbert-base-uncased"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_TASKS, CLS_PER_TASK = 5, 4
N_CLASSES = N_TASKS * CLS_PER_TASK
PER_CLASS_TR, PER_CLASS_TE = 40, 20
MAXLEN, BATCH = 64, 32
ROUTER_REPLAY = 32                               # cached feats/region for the router

_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(MODEL)
    return _TOK


def load_tasks(seed: int):
    """20NG → N_TASKS tasks of CLS_PER_TASK classes; subsample per class per seed."""
    g = torch.Generator().manual_seed(seed)
    out = {"train": [], "test": []}
    for split, per_class in (("train", PER_CLASS_TR), ("test", PER_CLASS_TE)):
        raw = fetch_20newsgroups(subset=split, remove=("headers", "footers", "quotes"))
        texts, ys = [], []
        y = torch.tensor(raw.target)
        for c in range(N_CLASSES):
            idx = (y == c).nonzero().flatten()
            idx = idx[torch.randperm(len(idx), generator=g)[:per_class]]
            texts += [raw.data[i] for i in idx.tolist()]
            ys += [c] * len(idx)
        enc = tok()(texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=MAXLEN)
        out[split] = {"ids": enc["input_ids"], "mask": enc["attention_mask"],
                      "y": torch.tensor(ys)}
    # slice by task
    def task_slice(d, t):
        lo, hi = t * CLS_PER_TASK, (t + 1) * CLS_PER_TASK
        m = (d["y"] >= lo) & (d["y"] < hi)
        return {"ids": d["ids"][m], "mask": d["mask"][m], "y": d["y"][m]}
    return [{"classes": tuple(range(t * CLS_PER_TASK, (t + 1) * CLS_PER_TASK)),
             "train": task_slice(out["train"], t),
             "test": task_slice(out["test"], t)} for t in range(N_TASKS)]


@torch.no_grad()
def cls_feats(model, ids, mask):
    """[CLS] of the last hidden state, batched — the pooled feature."""
    feats = []
    for i in range(0, len(ids), BATCH):
        h = model(input_ids=ids[i:i + BATCH].to(DEVICE),
                  attention_mask=mask[i:i + BATCH].to(DEVICE)).last_hidden_state
        feats.append(h[:, 0].cpu())
    return torch.cat(feats)


def lora_cfg():
    return LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                      target_modules=["q_lin", "v_lin"], task_type="FEATURE_EXTRACTION")


# ── Learners ─────────────────────────────────────────────────────────────────
class RegionLoRA:
    """per_region: peft multi-adapter, one LoRA + head per geometrically-routed
    region. Only the routed region's params train."""
    def __init__(self, base, base_feat_fn, region_radius=0.8, lr=2e-3, epochs=12):
        self.pm = None                           # created at first spawn
        self.base = base
        self.feat = base_feat_fn                 # routing features (frozen, no LoRA)
        self.radius, self.lr, self.epochs = region_radius, lr, epochs
        self.regions = []                        # {c, name, head, opt}

    def _route(self, h):
        if not self.regions:
            return None
        d = torch.stack([((r["c"] - h) ** 2).sum() for r in self.regions])
        j = int(d.argmin())
        return j if d[j].sqrt() <= self.radius else None

    def _nearest_centroid(self, h):
        return int(torch.stack([((r["c"] - h) ** 2).sum() for r in self.regions]).argmin())

    def _route_disc(self, h):
        """Argmax of the per-region linear rows — the learned router."""
        W = torch.stack([r["w"] for r in self.regions])
        b = torch.stack([r["b"] for r in self.regions])
        return int((W @ F.normalize(h, dim=0) + b).argmax())

    def _fit_router(self, iters: int = 300, lr: float = 0.05, wd: float = 5e-3):
        """Refit ALL region rows jointly on the whole replay buffer (regularized
        multinomial logistic on unit-norm features). Refitting does not forget —
        the replay buffer retains every region's features; the buffer IS the
        anti-forgetting device. Frozen-per-row gave incomparable, miscalibrated
        boundaries, so we refit jointly instead."""
        feats, labels = [], []
        for k, rr in enumerate(self.regions):
            if rr["replay"] is not None:
                feats.append(F.normalize(rr["replay"], dim=-1))
                labels.append(torch.full((len(rr["replay"]),), k, dtype=torch.long))
        X, y = torch.cat(feats), torch.cat(labels)
        for rr in self.regions:
            rr["w"].requires_grad_(True)
            rr["b"].requires_grad_(True)
        params = [p for rr in self.regions for p in (rr["w"], rr["b"])]
        opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        for _ in range(iters):
            W = torch.stack([rr["w"] for rr in self.regions])
            b = torch.stack([rr["b"] for rr in self.regions])
            opt.zero_grad()
            F.cross_entropy(X @ W.T + b, y).backward()
            opt.step()
        for rr in self.regions:
            rr["w"].requires_grad_(False)
            rr["b"].requires_grad_(False)

    def _spawn(self, c):
        name = f"r{len(self.regions)}"
        if self.pm is None:
            self.pm = get_peft_model(self.base, lora_cfg(), adapter_name=name).to(DEVICE)
        else:
            self.pm.add_adapter(name, lora_cfg())
        self.pm.set_adapter(name)
        head = nn.Linear(768, N_CLASSES).to(DEVICE)
        lora_params = [p for p in self.pm.parameters() if p.requires_grad]
        opt = torch.optim.Adam(lora_params + list(head.parameters()), lr=self.lr)
        self.regions.append({"c": c.detach().clone(), "name": name,
                             "head": head, "opt": opt, "classes": set(),
                             "replay": None, "w": torch.zeros(768), "b": torch.zeros(())})
        return len(self.regions) - 1

    def train_phase(self, task):
        d = task["train"]
        hbase = self.feat(d["ids"], d["mask"])   # frozen routing features
        r = self._route(hbase.mean(0))
        if r is None:
            r = self._spawn(hbase.mean(0))
        reg = self.regions[r]
        reg["classes"] |= set(d["y"].tolist())
        reg["replay"] = hbase[torch.randperm(len(hbase))[:ROUTER_REPLAY]].clone()
        self._fit_router()                        # refit all router rows on the buffer
        self.pm.set_adapter(reg["name"])          # activate this region's LoRA
        for _ in range(self.epochs):
            for i in range(0, len(d["y"]), BATCH):
                ids, mask = d["ids"][i:i + BATCH].to(DEVICE), d["mask"][i:i + BATCH].to(DEVICE)
                y = d["y"][i:i + BATCH].to(DEVICE)
                h = self.pm(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
                reg["opt"].zero_grad()
                F.cross_entropy(reg["head"](h), y).backward()
                reg["opt"].step()

    @torch.no_grad()
    def predict(self, task, route: str = "disc"):
        """route: "disc" learned modular router | "centroid" nearest centroid |
        "oracle" true-label region (upper bound isolating memory from routing)."""
        d = task["test"]
        hbase = self.feat(d["ids"], d["mask"])
        out = torch.full((len(d["y"]),), -1)
        by_region = {}
        for i in range(len(d["y"])):
            if route == "oracle":
                j = next((k for k, r in enumerate(self.regions)
                          if int(d["y"][i]) in r["classes"]), None)
            elif route == "disc":
                j = self._route_disc(hbase[i])
            else:
                j = self._nearest_centroid(hbase[i])
            if j is None:
                j = self._nearest_centroid(hbase[i])
            by_region.setdefault(j, []).append(i)
        for j, idxs in by_region.items():
            reg = self.regions[j]
            self.pm.set_adapter(reg["name"])
            mask_vec = torch.full((N_CLASSES,), float("-inf"), device=DEVICE)
            for c in reg["classes"]:
                mask_vec[c] = 0.0
            for b in range(0, len(idxs), BATCH):
                sel = idxs[b:b + BATCH]
                ids = d["ids"][sel].to(DEVICE)
                mask = d["mask"][sel].to(DEVICE)
                h = self.pm(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
                pred = (reg["head"](h) + mask_vec).argmax(-1).cpu()
                for k, s in enumerate(sel):
                    out[s] = pred[k]
        return out


class SharedLoRA:
    """shared: one LoRA adapter + one growing head across the whole stream."""
    def __init__(self, base, lr=2e-3, epochs=12):
        self.pm = get_peft_model(base, lora_cfg(), adapter_name="shared").to(DEVICE)
        self.head = nn.Linear(768, N_CLASSES).to(DEVICE)
        self.opt = torch.optim.Adam(
            [p for p in self.pm.parameters() if p.requires_grad] + list(self.head.parameters()),
            lr=lr)
        self.epochs, self.classes = epochs, set()

    def train_phase(self, task):
        d = task["train"]
        self.classes |= set(d["y"].tolist())
        for _ in range(self.epochs):
            for i in range(0, len(d["y"]), BATCH):
                ids, mask = d["ids"][i:i + BATCH].to(DEVICE), d["mask"][i:i + BATCH].to(DEVICE)
                y = d["y"][i:i + BATCH].to(DEVICE)
                h = self.pm(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
                self.opt.zero_grad()
                F.cross_entropy(self.head(h), y).backward()
                self.opt.step()

    @torch.no_grad()
    def predict(self, task):
        d = task["test"]
        mask_vec = torch.full((N_CLASSES,), float("-inf"), device=DEVICE)
        for c in self.classes:
            mask_vec[c] = 0.0
        out = []
        for i in range(0, len(d["y"]), BATCH):
            ids, mask = d["ids"][i:i + BATCH].to(DEVICE), d["mask"][i:i + BATCH].to(DEVICE)
            h = self.pm(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            out.append((self.head(h) + mask_vec).argmax(-1).cpu())
        return torch.cat(out)


class FullFT:
    """full_ft: unfreeze the whole backbone + one growing head."""
    def __init__(self, base, lr=2e-5, epochs=4):
        self.bb = base
        for p in self.bb.parameters():
            p.requires_grad_(True)
        self.head = nn.Linear(768, N_CLASSES).to(DEVICE)
        self.opt = torch.optim.Adam(list(self.bb.parameters()) + list(self.head.parameters()), lr=lr)
        self.epochs, self.classes = epochs, set()

    def train_phase(self, task):
        d = task["train"]
        self.classes |= set(d["y"].tolist())
        for _ in range(self.epochs):
            for i in range(0, len(d["y"]), BATCH):
                ids, mask = d["ids"][i:i + BATCH].to(DEVICE), d["mask"][i:i + BATCH].to(DEVICE)
                y = d["y"][i:i + BATCH].to(DEVICE)
                h = self.bb(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
                self.opt.zero_grad()
                F.cross_entropy(self.head(h), y).backward()
                self.opt.step()

    @torch.no_grad()
    def predict(self, task):
        d = task["test"]
        mask_vec = torch.full((N_CLASSES,), float("-inf"), device=DEVICE)
        for c in self.classes:
            mask_vec[c] = 0.0
        out = []
        for i in range(0, len(d["y"]), BATCH):
            ids, mask = d["ids"][i:i + BATCH].to(DEVICE), d["mask"][i:i + BATCH].to(DEVICE)
            h = self.bb(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            out.append((self.head(h) + mask_vec).argmax(-1).cpu())
        return torch.cat(out)


def accuracy(model, task) -> float:
    return (model.predict(task) == task["test"]["y"]).float().mean().item()


def make_model(kind: str, epochs: int):
    base = AutoModel.from_pretrained(MODEL).to(DEVICE)
    if kind == "full_ft":
        return FullFT(base, epochs=epochs)
    for p in base.parameters():
        p.requires_grad_(False)
    if kind == "shared":
        return SharedLoRA(base, epochs=epochs)
    frozen = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()   # separate clean routing net
    for p in frozen.parameters():
        p.requires_grad_(False)
    feat = lambda ids, mask: cls_feats(frozen, ids, mask)
    return RegionLoRA(base, feat, epochs=epochs)


def run_stream(kind: str, seed: int, epochs: int):
    torch.manual_seed(seed)
    tasks = load_tasks(seed)
    model = make_model(kind, epochs)
    rows = []
    for phase in range(N_TASKS):
        model.train_phase(tasks[phase])
        rows.append([accuracy(model, tasks[t]) for t in range(phase + 1)])
    n_regions = len(model.regions) if hasattr(model, "regions") else 0
    routing = None
    if hasattr(model, "regions"):                # decompose realized vs routing
        def finals(mode):
            a = [(model.predict(tasks[t], route=mode) == tasks[t]["test"]["y"]).float().mean().item()
                 for t in range(N_TASKS)]
            return {"task0": a[0], "all": sum(a) / len(a)}
        routing = {mode: finals(mode) for mode in ("disc", "centroid", "oracle")}
    del model
    gc.collect()
    return {"acc_matrix": rows, "n_regions": n_regions, "routing": routing}


def summarize(results):
    peak = [r["acc_matrix"][0][0] for r in results]
    final = [r["acc_matrix"][-1][0] for r in results]
    all_final = [sum(r["acc_matrix"][-1]) / len(r["acc_matrix"][-1]) for r in results]
    m = lambda xs: sum(xs) / len(xs)
    s = {"task0_peak": m(peak), "task0_final": m(final),
         "forgetting": m([p - f for p, f in zip(peak, final)]),
         "all_final": m(all_final), "n_regions": m([r["n_regions"] for r in results])}
    if results[0]["routing"] is not None:        # centroid baseline + oracle bound
        for mode in ("centroid", "oracle"):
            s[f"task0_{mode}"] = m([r["routing"][mode]["task0"] for r in results])
            s[f"all_{mode}"] = m([r["routing"][mode]["all"] for r in results])
    return s


def report(res):
    for k, r in res.items():
        line = (f"{k:11s} task0 {r['task0_peak']:.2f}→{r['task0_final']:.2f}  "
                f"forget {r['forgetting']:+.2f}  all_final {r['all_final']:.2f}  "
                f"regions {r['n_regions']:.1f}")
        if "all_oracle" in r:                    # per_region: disc | centroid | oracle
            line += (f"  | all: disc {r['all_final']:.2f}  "
                     f"centroid {r['all_centroid']:.2f}  oracle {r['all_oracle']:.2f}")
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="1 seed, quick sanity")
    ap.add_argument("--out", default="results/pretrained_cls.json")
    args = ap.parse_args()

    seeds = [0] if args.smoke else list(range(args.seeds))
    res = {k: summarize([run_stream(k, s, args.epochs) for s in seeds])
           for k in ("per_region", "shared", "full_ft")}
    report(res)
    if args.smoke:
        pr = res["per_region"]
        assert pr["all_final"] > res["shared"]["all_final"], "per_region did not beat shared LoRA"
        assert pr["task0_oracle"] >= pr["task0_peak"] - 0.05, \
            "oracle routing did not show the memory retains (gap is not routing)"
        print("smoke OK")
    if not args.smoke:
        with open(args.out, "w") as fh:
            json.dump({"config": {"model": MODEL, "seeds": seeds, "epochs": args.epochs,
                                  "n_tasks": N_TASKS, "cls_per_task": CLS_PER_TASK},
                       "summary": res}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
