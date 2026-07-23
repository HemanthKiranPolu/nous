"""
NOUS-CLS on a small open-weight LLM — sequential fine-tuning vs modular experts.

The scaling arc's LLM rung, at a size runnable on MPS. A real pretrained decoder
(`gpt2`, 125M) is fine-tuned on a stream of classification tasks, and we compare
STANDARD sequential fine-tuning (which forgets) against the modular per-task-expert
mechanism (which should retain):

  seq_lora : one shared LoRA adapter + growing head, LoRA-fine-tuned sequentially
             across tasks (the canonical continual-fine-tuning baseline).
  full_ft  : unfreeze the whole backbone + growing head, sequentially (worst case).
  modular  : one frozen-backbone LoRA expert + head per task, routed by geometry
             on the frozen last-token feature (per-class prototype, no task id).
             Only the current task's expert trains — prior experts are frozen.

Task stream: DBpedia-14 super-topics (org / people / place / nature / works),
5 tasks in phases. After EACH phase we measure accuracy on EVERY task seen so far
— the forgetting curve — and report average forgetting (mean drop from each task's
peak to its final) and final average accuracy.

Downloads once, cached: gpt2 (~500MB), DBpedia-14 (~150MB slice). MPS/CPU, ~minutes.

# ponytail: LoRA on c_attn only, tiny data subset, few epochs, 2 seeds — enough to
# show the forgetting gap on a real LLM, not a leaderboard run.
# ponytail: routing feature = frozen gpt2 last-token state (no LoRA), per-class
# prototype — the same near-optimal router the encoder experiments settled on.
"""

import argparse
import gc
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "gpt2"
D_FEAT = 768
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
GROUPS = [[0, 1], [2, 3, 4], [5, 6, 7, 8], [9, 10], [11, 12, 13]]   # DBpedia super-types
CLS2TASK = {c: ti for ti, cs in enumerate(GROUPS) for c in cs}
N_TASKS, N_CLASSES = len(GROUPS), 14
PER_CLASS_TR, PER_CLASS_TE = 40, 20
MAXLEN, BATCH = 48, 16

_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(MODEL)
        _TOK.pad_token = _TOK.eos_token
    return _TOK


def load_tasks(seed: int):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for split, per_class in (("train", PER_CLASS_TR), ("test", PER_CLASS_TE)):
        ds = load_dataset("fancyzhx/dbpedia_14", split=split)
        lab = torch.tensor(ds["label"])
        texts, ys = [], []
        for c in CLS2TASK:
            idx = (lab == c).nonzero().flatten()
            idx = idx[torch.randperm(len(idx), generator=g)[:per_class]]
            texts += [ds[int(i)]["content"] for i in idx]
            ys += [c] * len(idx)
        enc = tok()(texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=MAXLEN)
        out[split] = {"ids": enc["input_ids"], "mask": enc["attention_mask"],
                      "y": torch.tensor(ys)}
    def slc(d, t):
        m = torch.tensor([CLS2TASK[int(c)] == t for c in d["y"]])
        return {"ids": d["ids"][m], "mask": d["mask"][m], "y": d["y"][m]}
    return [{"train": slc(out["train"], t), "test": slc(out["test"], t)}
            for t in range(N_TASKS)]


def pool(hidden, mask):
    """Masked mean over tokens — gpt2's last-token state routes DBpedia at only
    ~0.38, mean-pool at ~0.71 (a decoder LM is a weaker sentence encoder than a
    trained embedder, so we take the better pooling)."""
    mm = mask.unsqueeze(-1).float()
    return (hidden * mm).sum(1) / mm.sum(1)


@torch.no_grad()
def feats(model, ids, mask):
    out = []
    for i in range(0, len(ids), BATCH):
        mb = mask[i:i + BATCH].to(DEVICE)
        h = model(input_ids=ids[i:i + BATCH].to(DEVICE), attention_mask=mb).last_hidden_state
        out.append(F.normalize(pool(h, mb), dim=-1).cpu())
    return torch.cat(out)


def lora_cfg():
    return LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                      target_modules=["c_attn"], task_type="FEATURE_EXTRACTION")


# ── Learners ─────────────────────────────────────────────────────────────────
class Modular:
    """One frozen-backbone LoRA expert + head per task, geometric per-class-proto
    routing. Only the current task's expert trains; prior experts stay frozen."""
    def __init__(self, base, feat_fn, lr=1e-3, epochs=10):
        self.pm, self.base, self.feat = None, base, feat_fn
        self.lr, self.epochs, self.regions = lr, epochs, []

    def _proto_route(self, h):
        P, reg = [], []
        for k, r in enumerate(self.regions):
            P.append(r["protos"])
            reg += [k] * len(r["protos"])
        P = torch.cat(P)
        return reg[int(((P - h) ** 2).sum(-1).argmin())]

    def train_phase(self, task):
        d = task["train"]
        hbase = self.feat(d["ids"], d["mask"])
        name = f"t{len(self.regions)}"
        if self.pm is None:
            self.pm = get_peft_model(self.base, lora_cfg(), adapter_name=name).to(DEVICE)
        else:
            self.pm.add_adapter(name, lora_cfg())
        self.pm.set_adapter(name)
        head = nn.Linear(D_FEAT, N_CLASSES).to(DEVICE)
        opt = torch.optim.Adam([p for p in self.pm.parameters() if p.requires_grad]
                               + list(head.parameters()), lr=self.lr)
        cls = sorted(set(d["y"].tolist()))
        protos = torch.stack([hbase[d["y"] == c].mean(0) for c in cls])
        self.regions.append({"name": name, "head": head, "classes": set(cls), "protos": protos})
        for _ in range(self.epochs):
            for i in range(0, len(d["y"]), BATCH):
                ids, mask = d["ids"][i:i + BATCH].to(DEVICE), d["mask"][i:i + BATCH].to(DEVICE)
                y = d["y"][i:i + BATCH].to(DEVICE)
                h = pool(self.pm(input_ids=ids, attention_mask=mask).last_hidden_state, mask)
                opt.zero_grad()
                F.cross_entropy(head(h), y).backward()
                opt.step()

    @torch.no_grad()
    def predict(self, task):
        d = task["test"]
        hbase = self.feat(d["ids"], d["mask"])
        out = torch.full((len(d["y"]),), -1)
        by = {}
        for i in range(len(d["y"])):
            by.setdefault(self._proto_route(hbase[i]), []).append(i)
        for j, idxs in by.items():
            reg = self.regions[j]
            self.pm.set_adapter(reg["name"])
            mv = torch.full((N_CLASSES,), float("-inf"), device=DEVICE)
            for c in reg["classes"]:
                mv[c] = 0.0
            for b in range(0, len(idxs), BATCH):
                sel = idxs[b:b + BATCH]
                ids, mask = d["ids"][sel].to(DEVICE), d["mask"][sel].to(DEVICE)
                h = pool(self.pm(input_ids=ids, attention_mask=mask).last_hidden_state, mask)
                pred = (reg["head"](h) + mv).argmax(-1).cpu()
                for k, s in enumerate(sel):
                    out[s] = pred[k]
        return out


class Sequential:
    """Baseline: ONE shared LoRA adapter + one growing head, fine-tuned on each
    task in turn (standard continual fine-tuning). full=True unfreezes the backbone."""
    def __init__(self, base, lr=1e-3, epochs=10, full=False):
        self.full = full
        if full:
            self.net = base
            for p in self.net.parameters():
                p.requires_grad_(True)
            params = list(self.net.parameters())
        else:
            self.net = get_peft_model(base, lora_cfg(), adapter_name="shared").to(DEVICE)
            params = [p for p in self.net.parameters() if p.requires_grad]
        self.net = self.net.to(DEVICE)
        self.head = nn.Linear(D_FEAT, N_CLASSES).to(DEVICE)
        self.opt = torch.optim.Adam(params + list(self.head.parameters()),
                                    lr=2e-5 if full else lr)
        self.epochs, self.classes = epochs, set()

    def train_phase(self, task):
        d = task["train"]
        self.classes |= set(d["y"].tolist())
        for _ in range(self.epochs):
            for i in range(0, len(d["y"]), BATCH):
                ids, mask = d["ids"][i:i + BATCH].to(DEVICE), d["mask"][i:i + BATCH].to(DEVICE)
                y = d["y"][i:i + BATCH].to(DEVICE)
                h = pool(self.net(input_ids=ids, attention_mask=mask).last_hidden_state, mask)
                self.opt.zero_grad()
                F.cross_entropy(self.head(h), y).backward()
                self.opt.step()

    @torch.no_grad()
    def predict(self, task):
        d = task["test"]
        mv = torch.full((N_CLASSES,), float("-inf"), device=DEVICE)
        for c in self.classes:
            mv[c] = 0.0
        out = []
        for i in range(0, len(d["y"]), BATCH):
            ids, mask = d["ids"][i:i + BATCH].to(DEVICE), d["mask"][i:i + BATCH].to(DEVICE)
            h = pool(self.net(input_ids=ids, attention_mask=mask).last_hidden_state, mask)
            out.append((self.head(h) + mv).argmax(-1).cpu())
        return torch.cat(out)


def accuracy(model, task):
    return (model.predict(task) == task["test"]["y"]).float().mean().item()


def make_model(kind, epochs):
    base = AutoModel.from_pretrained(MODEL).to(DEVICE)
    if kind == "full_ft":
        return Sequential(base, epochs=epochs, full=True)
    for p in base.parameters():
        p.requires_grad_(False)
    if kind == "seq_lora":
        return Sequential(base, epochs=epochs)
    frozen = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()   # clean routing net
    for p in frozen.parameters():
        p.requires_grad_(False)
    return Modular(base, lambda ids, m: feats(frozen, ids, m), epochs=epochs)


def run_stream(kind, seed, epochs):
    torch.manual_seed(seed)
    tasks = load_tasks(seed)
    model = make_model(kind, epochs)
    rows = []                                    # rows[k][t] = acc on task t after phase k
    for phase in range(N_TASKS):
        model.train_phase(tasks[phase])
        rows.append([accuracy(model, tasks[t]) for t in range(phase + 1)])
    n_regions = len(model.regions) if hasattr(model, "regions") else 0
    del model
    gc.collect()
    return {"acc_matrix": rows, "n_regions": n_regions}


def metrics(results):
    """Average forgetting (mean peak→final drop over prior tasks) + final avg acc."""
    final_avg, forget = [], []
    for r in results:
        M = r["acc_matrix"]
        last = M[-1]
        final_avg.append(sum(last) / len(last))
        drops = []
        for t in range(N_TASKS - 1):                # tasks that have a later phase
            peak = max(M[p][t] for p in range(t, N_TASKS))
            drops.append(peak - M[-1][t])
        forget.append(sum(drops) / len(drops))
    m = lambda xs: sum(xs) / len(xs)
    return {"final_avg": m(final_avg), "forgetting": m(forget),
            "task0_final": m([r["acc_matrix"][-1][0] for r in results]),
            "n_regions": m([r["n_regions"] for r in results])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/llm_cl.json")
    args = ap.parse_args()

    seeds = [0] if args.smoke else list(range(args.seeds))
    out = {"config": {"model": MODEL, "seeds": seeds, "epochs": args.epochs,
                      "n_tasks": N_TASKS}, "summary": {}}
    for kind in ("modular", "seq_lora", "full_ft"):
        res = [run_stream(kind, s, args.epochs) for s in seeds]
        out["summary"][kind] = metrics(res)
        r = out["summary"][kind]
        print(f"{kind:9s} task0_final {r['task0_final']:.2f}  final_avg {r['final_avg']:.2f}  "
              f"avg_forget {r['forgetting']:+.2f}  regions {r['n_regions']:.0f}")
    if args.smoke:
        assert out["summary"]["modular"]["forgetting"] < out["summary"]["seq_lora"]["forgetting"], \
            "modular did not forget less than sequential LoRA fine-tuning"
        print("smoke OK")
    else:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
