"""
NOUS-CLS on a transformer — "move from toy" (step 4).

Lifts the mod-5 continual result onto a real gradient-trained net: a FROZEN
transformer backbone with per-REGION LoRA-style experts, routed by geometry on
the backbone's pooled feature (no task label), compared to a single shared
adapter and to full fine-tuning.

Task stream: Split-digits (sklearn `load_digits`, 8×8, zero download) → 5 binary
tasks {0,1},{2,3},…,{8,9} presented in phases. After each phase, accuracy on all
tasks seen so far → retention / forgetting.

Learners:
  per_region : one low-rank adapter + head per discovered region; only the routed
               region trains. Test-time routing is geometric (nearest centroid),
               so no task id is needed at inference — the step-3 mechanism.
  shared     : one adapter + one growing head, trained through the whole stream.
  full_ft    : unfreeze the whole backbone + one growing head (upper-bound forget).

Claim: task-conditioned adapters retain prior tasks where a shared adapter and
full fine-tuning forget — now on a real transformer whose features are learned.

Runs on CPU in ~1–2 min.

# ponytail: adapter is a low-rank residual on the POOLED feature, not LoRA
# injected into every attention matrix — same local-vs-shared plasticity test,
# far less plumbing. Injecting into q/k/v/o is the faithful upgrade.
# ponytail: experts spawn at task boundaries (one per phase), not on per-example
# surprise as in the toy. Test-time routing is still label-free. Surprise-spawn
# is the upgrade that removes the last boundary crutch.
"""

import argparse
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

DEVICE = "cpu"                                   # tiny model — CPU is fastest here
N_TASKS = 5
D_MODEL, NHEAD, NLAYERS, DFF = 32, 2, 2, 64
SEQ, TOK = 8, 8                                  # 8×8 image → 8 tokens of dim 8
N_CLASSES = 2 * N_TASKS


def load_tasks(seed: int):
    d = load_digits()
    X = torch.tensor(d.images, dtype=torch.float32).reshape(-1, SEQ, TOK) / 16.0
    y = torch.tensor(d.target)
    idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    X, y = X[idx], y[idx]
    ntr = int(0.8 * len(X))
    (Xtr, ytr), (Xte, yte) = (X[:ntr], y[:ntr]), (X[ntr:], y[ntr:])
    tasks = []
    for t in range(N_TASKS):
        c = (2 * t, 2 * t + 1)
        m_tr = (ytr == c[0]) | (ytr == c[1])
        m_te = (yte == c[0]) | (yte == c[1])
        tasks.append({"classes": c,
                      "train": (Xtr[m_tr], ytr[m_tr]),
                      "test": (Xte[m_te], yte[m_te])})
    return tasks


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(TOK, D_MODEL)
        self.pos = nn.Parameter(torch.randn(SEQ, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, DFF, batch_first=True,
                                           dropout=0.0)
        self.tf = nn.TransformerEncoder(layer, NLAYERS)

    def forward(self, x):                        # x: (B, SEQ, TOK)
        h = self.embed(x) + self.pos
        return self.tf(h).mean(1)                # (B, D_MODEL) pooled feature


def pretrain_backbone(backbone: nn.Module, tasks, seed: int, iters: int = 300):
    """Briefly supervised-pretrain on ALL digits, then the caller freezes it —
    the 'use a pretrained backbone' analog. Gives non-trivial frozen features."""
    torch.manual_seed(seed)
    head = nn.Linear(D_MODEL, N_CLASSES)
    X = torch.cat([t["train"][0] for t in tasks])
    y = torch.cat([t["train"][1] for t in tasks])
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)
    for _ in range(iters):
        opt.zero_grad()
        F.cross_entropy(head(backbone(X)), y).backward()
        opt.step()


class Expert(nn.Module):
    """A low-rank residual adapter on the pooled feature + a class head."""
    def __init__(self, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(D_MODEL, rank)
        self.up = nn.Linear(rank, D_MODEL)
        nn.init.zeros_(self.up.weight)           # adapter starts at identity
        nn.init.zeros_(self.up.bias)
        self.head = nn.Linear(D_MODEL, N_CLASSES)
        self.classes = set()

    def forward(self, h):
        h = h + self.up(F.gelu(self.down(h)))
        return self.head(h)


# ── Learners ─────────────────────────────────────────────────────────────────
class RegionExperts:
    """per_region: geometric router over frozen features; one Expert per region.
    Only the routed region trains, so old regions' params are never touched."""
    def __init__(self, backbone, region_radius: float = 6.0, lr: float = 1e-2):
        self.bb = backbone
        self.radius, self.lr = region_radius, lr
        self.regions = []                        # each: {c, expert, opt}

    def _feat(self, x):
        with torch.no_grad():
            return self.bb(x)

    def _route(self, h):                         # h: (D,) single feature
        if not self.regions:
            return None
        d = torch.stack([((r["c"] - h) ** 2).sum() for r in self.regions])
        j = int(d.argmin())
        return j if d[j].sqrt() <= self.radius else None

    def _spawn(self, c):
        e = Expert()
        self.regions.append({"c": c.detach().clone(), "expert": e,
                             "opt": torch.optim.Adam(e.parameters(), lr=self.lr)})
        return len(self.regions) - 1

    def train_phase(self, data, epochs: int):
        X, y = data
        H = self._feat(X)                        # frozen features for the whole task
        r = self._route(H.mean(0))               # one region per task (boundary spawn)
        if r is None:
            r = self._spawn(H.mean(0))
        reg = self.regions[r]
        reg["expert"].classes |= set(y.tolist())
        for _ in range(epochs):
            reg["opt"].zero_grad()
            F.cross_entropy(reg["expert"](H), y).backward()
            reg["opt"].step()

    def predict(self, x):                        # geometric test-time routing
        H = self._feat(x)
        out = torch.full((len(x),), -1)
        for i in range(len(x)):
            j = self._route(H[i])
            if j is None:
                j = int(torch.stack([((r["c"] - H[i]) ** 2).sum()
                                     for r in self.regions]).argmin())
            reg = self.regions[j]
            logit = reg["expert"](H[i:i + 1])[0]
            mask = torch.full((N_CLASSES,), float("-inf"))
            for c in reg["expert"].classes:
                mask[c] = 0.0
            out[i] = int((logit + mask).argmax())
        return out


class SharedAdapter:
    """shared: one adapter + one growing head trained through the whole stream."""
    def __init__(self, backbone, lr: float = 1e-2):
        self.bb = backbone
        self.expert = Expert()
        self.opt = torch.optim.Adam(self.expert.parameters(), lr=lr)

    def train_phase(self, data, epochs: int):
        X, y = data
        with torch.no_grad():
            H = self.bb(X)
        self.expert.classes |= set(y.tolist())
        for _ in range(epochs):
            self.opt.zero_grad()
            F.cross_entropy(self.expert(H), y).backward()
            self.opt.step()

    def predict(self, x):
        with torch.no_grad():
            logit = self.expert(self.bb(x))
        mask = torch.full((N_CLASSES,), float("-inf"))
        for c in self.expert.classes:
            mask[c] = 0.0
        return (logit + mask).argmax(-1)


class FullFinetune:
    """full_ft: unfreeze the backbone + one growing head — upper-bound forgetting."""
    def __init__(self, backbone, lr: float = 1e-3):
        self.bb = backbone
        for p in self.bb.parameters():
            p.requires_grad_(True)
        self.head = nn.Linear(D_MODEL, N_CLASSES)
        self.classes = set()
        self.opt = torch.optim.Adam(list(self.bb.parameters()) + list(self.head.parameters()), lr=lr)

    def train_phase(self, data, epochs: int):
        X, y = data
        self.classes |= set(y.tolist())
        for _ in range(epochs):
            self.opt.zero_grad()
            F.cross_entropy(self.head(self.bb(X)), y).backward()
            self.opt.step()

    def predict(self, x):
        with torch.no_grad():
            logit = self.head(self.bb(x))
        mask = torch.full((N_CLASSES,), float("-inf"))
        for c in self.classes:
            mask[c] = 0.0
        return (logit + mask).argmax(-1)


def accuracy(model, task) -> float:
    X, y = task["test"]
    return (model.predict(X) == y).float().mean().item()


def make_model(kind: str, seed: int, tasks):
    torch.manual_seed(seed)
    bb = Backbone()
    pretrain_backbone(bb, tasks, seed)           # pretrain then (for adapters) freeze
    if kind == "full_ft":
        return FullFinetune(bb)
    for p in bb.parameters():
        p.requires_grad_(False)
    return RegionExperts(bb) if kind == "per_region" else SharedAdapter(bb)


def run_stream(kind: str, seed: int, epochs: int):
    tasks = load_tasks(seed)
    model = make_model(kind, seed, tasks)
    rows = []                                    # rows[k][t] = acc on task t after phase k
    for phase in range(N_TASKS):
        model.train_phase(tasks[phase]["train"], epochs)
        rows.append([accuracy(model, tasks[t]) for t in range(phase + 1)])
    n_regions = len(model.regions) if hasattr(model, "regions") else 0
    return {"acc_matrix": rows, "n_regions": n_regions}


def summarize(results):
    peak = [r["acc_matrix"][0][0] for r in results]          # task0 acc after phase 0
    final = [r["acc_matrix"][-1][0] for r in results]        # task0 acc after all phases
    all_final = [_mean(r["acc_matrix"][-1]) for r in results]
    return {"task0_peak": _mean(peak), "task0_final": _mean(final),
            "forgetting": _mean([p - f for p, f in zip(peak, final)]),
            "all_final": _mean(all_final),
            "n_regions": _mean([r["n_regions"] for r in results])}


def _mean(xs):
    return sum(xs) / len(xs)


def selfcheck():
    kinds = ("per_region", "shared", "full_ft")
    res = {k: summarize([run_stream(k, s, 60) for s in range(3)]) for k in kinds}
    for k in kinds:
        r = res[k]
        print(f"{k:11s} task0 {r['task0_peak']:.2f}→{r['task0_final']:.2f}  "
              f"forget {r['forgetting']:+.2f}  all_final {r['all_final']:.2f}  "
              f"regions {r['n_regions']:.1f}")
    assert res["per_region"]["task0_peak"] > 0.8, "per_region failed to learn task 0"
    assert res["per_region"]["forgetting"] < res["shared"]["forgetting"], \
        "per_region did not beat shared-adapter forgetting"
    assert res["per_region"]["forgetting"] < res["full_ft"]["forgetting"], \
        "per_region did not beat full-finetune forgetting"
    print("selfcheck OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="results/transformer_cls.json")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    seeds = list(range(args.seeds))
    out = {"config": {"seeds": seeds, "epochs": args.epochs, "n_tasks": N_TASKS,
                      "d_model": D_MODEL, "layers": NLAYERS},
           "summary": {}}
    for kind in ("per_region", "shared", "full_ft"):
        res = [run_stream(kind, s, args.epochs) for s in seeds]
        out["summary"][kind] = summarize(res)
        r = out["summary"][kind]
        print(f"{kind:11s} task0 {r['task0_peak']:.3f}→{r['task0_final']:.3f}  "
              f"forget {r['forgetting']:+.3f}  all_final {r['all_final']:.3f}  "
              f"regions {r['n_regions']:.1f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
