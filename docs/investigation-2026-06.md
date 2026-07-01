# NOUS investigation — honest summary (June 2026)

A full, honest account of what was tested, what holds, and what doesn't. Written
so nobody (including future us) re-litigates settled questions.

## TL;DR
- The original headline ("NOUS 100% vs transformer 5.9% on novel compositions")
  is **unverifiable** — the script it cited (`novelty_test.py`) does not exist
  in the repo.
- **One real, rigorous positive:** on a controlled compositional-generalization
  toy, NOUS (central, factored decode) beats param-matched from-scratch baselines
  — **57.5% vs MLP 12.5% vs Transformer 9.2%** held-out, 32 seeds, train-fit
  gated, non-overlapping CIs. Merged (PRs #1–#2).
- **Everything aimed at a real-world win came up empty**, with data (below).
- **The one genuinely novel asset** to emerge: **EV-TRM** — a tiny recursive
  reasoner with a jointly-trained energy head that is *calibrated to its own
  errors* (self-verification + abstention). PoC works (PR #6).

## What was tested and what happened

| Direction | Result | Verdict |
|---|---|---|
| Reproduce 100%/5.9% claim | script absent from repo | ✗ unverifiable |
| Controlled comp-gen toy (NOUS central) | 57.5% vs 12.5% / 9.2%, gated, CIs clear | ✅ real (toy) |
| Real SCAN, NOUS-as-latent | fits train, 0–1.9% held-out; **ablation identical** | ✗ inert |
| Real SCAN, pure single-`q*` decode | capacity wall, can't fit 50 examples | ✗ |
| NOUS as accept/reject **verifier** | valid-recall ≈ 0 = MLP | ✗ inert |
| LLM **controller** (route toy to NOUS) | 0%→78% on toy… | ⚠ but see next |
| …tested with a **real LLM** (Qwen-1.5B) | LLM **100%** on the same toy | ✗ no niche |
| NOUS-energy vs classical CSP (feasibility) | DSATUR/min-conf 100% & faster | ✗ loses |
| NOUS-energy vs SA/min-conf (MAX-CSP quality) | 3rd place, within ~10% | ✗ loses |
| Portfolio / algorithm-selection ceiling | oracle beats best single (SA) by **1%** | ✗ no room |
| **EV-TRM** (TRM core + energy head), 4×4 Sudoku | solve **95.8%**, error-flag **AUC 0.81** | ✅ novel |

## The two things that are actually true
1. **vs LLMs on constraint reasoning, energy/iterative methods win big** (LLM 0%,
   solvers ~100%). But there the *right tool is a classical solver*, not NOUS.
2. **EV-TRM's calibrated self-verification is a genuinely new capability** — a
   tiny reasoner that knows when it is wrong (AUC 0.81 → abstention 95.8%→98.3%).
   No classical solver or LLM provides this. This is also *why a standalone
   verifier was inert*: the verifier must be trained **jointly** with the
   generator, which the energy head is.

## What is NOT true (settled, do not retry)
- NOUS does **not** beat well-engineered classical CSP solvers (DSATUR /
  min-conflicts / SA) on speed **or** quality — four independent tests agree.
- NOUS's original **RBF energy + EqProp + single-`q*`** core is obsolete; it
  underperforms everything. Drop it.
- A **portfolio** of these solvers can't win: the best single (SA) is
  near-oracle (1% ceiling).
- NOUS-as-latent / NOUS-as-component behaves like a plain net (ablation-proven).

## Literature grounding (why EV-TRM is the right pivot)
The NOUS *idea* (non-autoregressive iterative energy minimization) is validated
by 2025 SOTA — but modern implementations, not NOUS's:
- Energy-Based Transformers — out-scale Transformer++ 35% [arXiv:2507.02092].
- Tiny Recursive Model (7M) — beats DeepSeek-R1 / Gemini-2.5-Pro on ARC/Sudoku
  [arXiv:2510.04871]; ~87% Sudoku-Extreme after ~20h/1-GPU.
- Kona EBM — 96% hard Sudoku vs ~2% for LLMs.
Caveat: deep supervision drives most TRM gains; these are trained **per task**.

## Recommendation
- **Stop** trying to out-solve classical CSP solvers, and stop reviving RBF/EqProp.
- **Pursue EV-TRM**: scale it to 9×9 Sudoku-Extreme on a persistent GPU (~20h;
  the synchronous-notebook path can't host it) and keep the energy head — the
  defensible story is *a self-verifying tiny recursive reasoner with calibrated
  abstention and an honest LLM trust-gate*, not "beats GPT / beats classical."
- Publish at that altitude: a mechanism + capability contribution, honestly scoped.

## Artifacts
PRs #1–#2 (comp-gen + CI, merged), #3 (SCAN scaffold, draft), #4 (latency),
#5 (controller study + real-LLM check), #6 (EV-TRM). Results under `results/`.
