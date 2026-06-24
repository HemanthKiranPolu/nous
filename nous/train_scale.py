"""
NOUS Scale Training — multi-dataset progressive pipeline.

Stages (run sequentially or pick one):
  --stage mnist-10k    : MNIST 10k subset, fast ODE baseline
  --stage mnist-full   : Full 60k MNIST, batched eval
  --stage wikitext     : WikiText-2 next-token prediction
  --stage cifar        : CIFAR-10 image classification
  --stage code         : Python code completion (HuggingFace codeparrot-clean)

Fast ODE settings for scale: n_steps=20, dt=0.2 (overdamped = fewer steps needed).
Accumulate=8 for stable gradients on larger datasets.

Usage:
  python -m nous.train_scale --stage mnist-10k
  python -m nous.train_scale --stage mnist-full --epochs 10
  python -m nous.train_scale --stage wikitext --epochs 5
  python -m nous.train_scale --stage cifar --epochs 20
"""

import argparse, os, sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--stage",  default="mnist-10k",
    choices=["mnist-10k","mnist-full","wikitext","cifar","code"])
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--lr",     type=float, default=1e-3)
parser.add_argument("--beta",   type=float, default=0.1)
parser.add_argument("--subset", type=int, default=None,
    help="Override dataset size (for quick tests)")
args = parser.parse_args()

OUT_DIR = f"nous_output_{args.stage.replace('-','_')}"
os.makedirs(OUT_DIR, exist_ok=True)

from nous.energy_net import EnergyNet
from nous.el_solver import EulerLagrangeSolver
from nous.eqprop_centered import CenteredEqProp
from nous.annealing import AnnealingScheduler

torch.manual_seed(42)

# ── Stage configs ──────────────────────────────────────────────────────────────
STAGE_CFG = {
    "mnist-10k": dict(
        n_train=10000, n_test=2000, epochs=20,
        embed_dim=64,  state_dim=64,  hidden=128, depth=3, n_rbf=32,
        n_steps=20, dt=0.2, accumulate=8, grad_clip=1.0,
        task="mnist",
    ),
    "mnist-full": dict(
        n_train=60000, n_test=10000, epochs=10,
        embed_dim=64,  state_dim=64,  hidden=128, depth=3, n_rbf=32,
        n_steps=20, dt=0.2, accumulate=16, grad_clip=1.0,
        task="mnist",
    ),
    "wikitext": dict(
        n_train=50000, n_test=5000, epochs=10,
        embed_dim=128, state_dim=128, hidden=256, depth=4, n_rbf=32,
        n_steps=20, dt=0.2, accumulate=16, grad_clip=1.0,
        task="wikitext",
    ),
    "cifar": dict(
        n_train=50000, n_test=10000, epochs=30,
        embed_dim=128, state_dim=128, hidden=256, depth=4, n_rbf=64,
        n_steps=20, dt=0.2, accumulate=16, grad_clip=1.0,
        task="cifar",
    ),
    "code": dict(
        n_train=20000, n_test=2000, epochs=10,
        embed_dim=128, state_dim=128, hidden=256, depth=4, n_rbf=32,
        n_steps=20, dt=0.2, accumulate=16, grad_clip=1.0,
        task="code",
    ),
}

cfg = STAGE_CFG[args.stage]
if args.epochs: cfg["epochs"] = args.epochs
if args.subset: cfg["n_train"] = args.subset

print(f"NOUS Scale — {args.stage}")
print(f"  Train: {cfg['n_train']}  |  Epochs: {cfg['epochs']}")
print(f"  State: {cfg['state_dim']}D  |  n_rbf: {cfg['n_rbf']}")
print(f"  β={args.beta}  accumulate={cfg['accumulate']}  n_steps={cfg['n_steps']}")
print()

# ── Dataset loading ───────────────────────────────────────────────────────────
task = cfg["task"]

if task == "mnist":
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    tr_full = datasets.MNIST("~/.cache/mnist", train=True, download=True, transform=tf)
    te_full = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tf)
    n_classes = 10
    input_dim = 784

    if cfg["n_train"] < 60000:
        per = cfg["n_train"] // n_classes
        idx, c = [], {i: 0 for i in range(n_classes)}
        for i, (_, l) in enumerate(tr_full):
            if c[l] < per: idx.append(i); c[l] += 1
            if len(idx) >= cfg["n_train"]: break
        train_ds = torch.utils.data.Subset(tr_full, idx)
    else:
        train_ds = tr_full

    if cfg["n_test"] < 10000:
        per = cfg["n_test"] // n_classes
        ti, tc = [], {i: 0 for i in range(n_classes)}
        for i, (_, l) in enumerate(te_full):
            if tc[l] < per: ti.append(i); tc[l] += 1
            if len(ti) >= cfg["n_test"]: break
        test_ds = torch.utils.data.Subset(te_full, ti)
    else:
        test_ds = te_full

    print(f"  MNIST: {len(train_ds)} train / {len(test_ds)} test")

elif task == "cifar":
    from torchvision import datasets, transforms
    tf_tr = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    tf_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    train_ds = datasets.CIFAR10("~/.cache/cifar10", train=True, download=True, transform=tf_tr)
    test_ds  = datasets.CIFAR10("~/.cache/cifar10", train=False, download=True, transform=tf_te)
    n_classes = 10
    input_dim = 3072
    if args.subset:
        train_ds = torch.utils.data.Subset(train_ds, range(args.subset))
    print(f"  CIFAR-10: {len(train_ds)} train / {len(test_ds)} test")

elif task == "wikitext":
    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
    except ImportError:
        print("pip install datasets transformers  # required for wikitext stage")
        sys.exit(1)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    n_classes  = vocab_size
    SEQ_LEN    = 64

    print("Loading WikiText-2...")
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_wiki(split, max_tokens):
        tokens = []
        for row in wiki[split]["text"]:
            t = tokenizer.encode(row.strip())
            tokens.extend(t)
            if len(tokens) >= max_tokens: break
        return tokens[:max_tokens]

    train_tokens = tokenize_wiki("train", cfg["n_train"] + SEQ_LEN)
    test_tokens  = tokenize_wiki("validation", cfg["n_test"] + SEQ_LEN)

    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, tokens, seq_len):
            self.t, self.s = tokens, seq_len
        def __len__(self): return len(self.t) - self.s
        def __getitem__(self, i):
            ctx = torch.tensor(self.t[i:i+self.s], dtype=torch.long)
            nxt = self.t[i+self.s]
            return ctx, nxt

    train_ds = TokenDataset(train_tokens, SEQ_LEN)
    test_ds  = TokenDataset(test_tokens,  SEQ_LEN)
    input_dim = SEQ_LEN  # will embed each context window
    print(f"  WikiText-2: {len(train_ds)} train / {len(test_ds)} test")

elif task == "code":
    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
    except ImportError:
        print("pip install datasets transformers  # required for code stage")
        sys.exit(1)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    n_classes  = vocab_size
    SEQ_LEN    = 64

    print("Loading code dataset (codeparrot/codeparrot-clean-train, streaming)...")
    code_ds = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=True)

    code_tokens = []
    for sample in code_ds:
        t = tokenizer.encode(sample["content"])
        code_tokens.extend(t)
        if len(code_tokens) >= cfg["n_train"] + SEQ_LEN: break

    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, tokens, seq_len):
            self.t, self.s = tokens, seq_len
        def __len__(self): return len(self.t) - self.s
        def __getitem__(self, i):
            ctx = torch.tensor(self.t[i:i+self.s], dtype=torch.long)
            nxt = self.t[i+self.s]
            return ctx, nxt

    all_tokens = code_tokens[:cfg["n_train"] + cfg["n_test"] + SEQ_LEN]
    train_ds = TokenDataset(all_tokens[:cfg["n_train"] + SEQ_LEN], SEQ_LEN)
    test_ds  = TokenDataset(all_tokens[cfg["n_train"]:], SEQ_LEN)
    input_dim = SEQ_LEN
    print(f"  Code: {len(train_ds)} train / {len(test_ds)} test")

# ── Architecture ──────────────────────────────────────────────────────────────
EMBED_DIM  = cfg["embed_dim"]
STATE_DIM  = cfg["state_dim"]

if task in ("wikitext", "code"):
    # Token embedding: context window → dense embedding
    tok_embed  = nn.Embedding(vocab_size, EMBED_DIM)
    nn.init.normal_(tok_embed.weight, 0, 0.02)
    projector  = nn.Sequential(
        tok_embed,
        nn.AdaptiveAvgPool1d(1),   # mean-pool over sequence → (batch, embed)
    )
    def project(ctx):
        e = tok_embed(ctx)         # (seq, embed)
        return e.mean(0)           # (embed,)
else:
    projector = nn.Linear(input_dim, EMBED_DIM)
    nn.init.xavier_uniform_(projector.weight, gain=0.5)
    nn.init.zeros_(projector.bias)
    def project(x): return projector(x)

E = EnergyNet(input_dim=EMBED_DIM, state_dim=STATE_DIM,
              hidden=cfg["hidden"], depth=cfg["depth"], n_rbf=cfg["n_rbf"])
decoder = nn.Linear(STATE_DIM, n_classes)
nn.init.xavier_uniform_(decoder.weight, gain=0.3)
nn.init.zeros_(decoder.bias)

if task in ("wikitext","code"):
    all_params = list(tok_embed.parameters()) + list(E.parameters()) + list(decoder.parameters())
else:
    all_params = list(projector.parameters()) + list(E.parameters()) + list(decoder.parameters())

optimizer = torch.optim.Adam(all_params, lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
annealer  = AnnealingScheduler(beta_0=0.5, lambda_=0.0003, beta_max=8.0, alpha_0=args.lr)

solver  = EulerLagrangeSolver(E, dt=cfg["dt"], n_steps=cfg["n_steps"], delta=1e-3)
ceqprop = CenteredEqProp(E, solver, decoder, optimizer,
                          beta=args.beta,
                          grad_clip=cfg["grad_clip"],
                          accumulate=cfg["accumulate"])

# EMA
ema_params = {n: p.clone().detach() for n, p in E.named_parameters()}
ema_params.update({f"dec_{n}": p.clone().detach() for n, p in decoder.named_parameters()})
EMA_DECAY = 0.995

def update_ema():
    with torch.no_grad():
        for n, p in E.named_parameters():
            ema_params[n].mul_(EMA_DECAY).add_(p.data, alpha=1-EMA_DECAY)
        for n, p in decoder.named_parameters():
            ema_params[f"dec_{n}"].mul_(EMA_DECAY).add_(p.data, alpha=1-EMA_DECAY)

def swap_ema(restore_from=None):
    live = {}
    for n, p in E.named_parameters():
        live[n] = p.data.clone()
        p.data.copy_(ema_params[n] if restore_from is None else restore_from[n])
    for n, p in decoder.named_parameters():
        live[f"dec_{n}"] = p.data.clone()
        p.data.copy_(ema_params[f"dec_{n}"] if restore_from is None else restore_from[f"dec_{n}"])
    return live

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(ds, max_samples=2000, use_ema=True):
    live = swap_ema() if use_ema else None
    correct = total = 0
    n = min(len(ds), max_samples)
    for i in range(n):
        sample = ds[i]
        if task in ("wikitext","code"):
            ctx, lbl = sample
            x = project(ctx).detach()
        else:
            img, lbl = sample
            x = project(img).detach()
        lbl_i = lbl.item() if hasattr(lbl, 'item') else int(lbl)
        q0    = torch.zeros(STATE_DIM)
        q_star = solver.solve(x, q0)
        pred   = decoder(q_star).argmax().item()
        correct += (pred == lbl_i)
        total   += 1
    if use_ema: swap_ema(live)
    return correct / total * 100

def perplexity(ds, max_samples=500):
    live = swap_ema()
    total_loss = n = 0
    for i in range(min(len(ds), max_samples)):
        sample = ds[i]
        ctx, lbl = sample
        x = project(ctx).detach()
        lbl_i = lbl.item() if hasattr(lbl, 'item') else int(lbl)
        q0 = torch.zeros(STATE_DIM)
        q_star = solver.solve(x, q0)
        logits = decoder(q_star)
        total_loss += F.cross_entropy(logits.unsqueeze(0),
                                       torch.tensor([lbl_i])).item()
        n += 1
    swap_ema(live)
    return math.exp(total_loss / n)

# ── Training ──────────────────────────────────────────────────────────────────
loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)

history = {"acc": [], "loss": []}
best_acc = 0.0

print(f"{'Ep':>3}  {'Loss':>7}  {'Acc%':>6}  {'LR':>8}  {'T':>5}")
print("─" * 38)

for epoch in range(cfg["epochs"]):
    t0 = time.time()
    losses = []

    for batch in loader:
        if task in ("wikitext","code"):
            ctx, lbl = batch
            ctx, lbl = ctx.squeeze(0), lbl.squeeze(0)
            xp = project(ctx)
        else:
            img, lbl = batch
            img, lbl = img.squeeze(0), lbl.squeeze(0)
            xp = project(img)

        x = xp.detach()

        if task in ("wikitext","code"):
            loss, _, _, _ = ceqprop.step(x, lbl, x_with_grad=None)
        else:
            loss, _, _, _ = ceqprop.step(x, lbl, x_with_grad=xp)

        losses.append(loss)

    update_ema()
    scheduler.step()

    avg_loss = float(np.mean(losses))
    history["loss"].append(avg_loss)

    if task in ("wikitext","code"):
        acc = 100.0 - perplexity(test_ds, 300)  # proxy: negative perplexity
        ppl = perplexity(test_ds, 300)
        print(f"{epoch:3d}  {avg_loss:7.4f}  ppl={ppl:6.1f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)
        history["acc"].append(-ppl)
        best_acc = max(best_acc, -ppl)
    else:
        acc = evaluate(test_ds, max_samples=min(len(test_ds), 2000))
        history["acc"].append(acc)
        best_acc = max(best_acc, acc)
        print(f"{epoch:3d}  {avg_loss:7.4f}  {acc:6.1f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nBest: {best_acc:.2f}")
if task == "mnist":
    print("\n── vs baselines ──")
    print("  Logistic (1k train):   ~85%")
    print("  Logistic (10k train):  ~92%")
    print("  Logistic (60k train):  ~92%")
    print("  MLP (60k train):       ~98%")
    print("  C-EP lit (60k):        ~99.6%")
    print(f"  NOUS-Scale:            {best_acc:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["loss"], marker="o", ms=3, lw=1.5, color="#9060ff")
axes[0].set_title("Loss"); axes[0].grid(alpha=0.3)
axes[1].plot(history["acc"],  marker="o", ms=3, lw=2.0, color="#60d0a0")
metric = "Test Acc (%)" if task in ("mnist","cifar") else "-Perplexity"
axes[1].set_title(metric); axes[1].grid(alpha=0.3)
plt.suptitle(f"NOUS-Scale {args.stage} | best={best_acc:.1f}", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/nous_scale_{args.stage}.png", dpi=130)
plt.close()
print(f"Plot: {OUT_DIR}/nous_scale_{args.stage}.png")
