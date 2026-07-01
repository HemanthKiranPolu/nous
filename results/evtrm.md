# EV-TRM — Energy-Verified Tiny Recursive Reasoner

A tiny recursive reasoner (TRM-style) + a jointly-trained **energy/verification
head**. The energy head adds what TRM/HRM lack: **calibrated self-verification**
→ abstention and a trust-gate for routing from an LLM. Source: `nous/evtrm.py`.

## Why this direction (literature)
The NOUS idea — non-autoregressive iterative energy minimization — is validated
by 2025 work, but NOUS's old RBF+EqProp core is obsolete. The winners:
- Energy-Based Transformers — out-scale Transformer++ 35%, +29% System-2 [arXiv:2507.02092].
- Tiny Recursive Model (TRM) — 7M params, 45% ARC-AGI-1, **beats** DeepSeek-R1 /
  Gemini-2.5-Pro; ~87% Sudoku-Extreme after ~20h/1-GPU [arXiv:2510.04871].
- Kona EBM — 96% hard Sudoku vs ~2% for LLMs.

Caveats from that literature: deep supervision (not recursion per se) drives most
gains; these are trained **per task**, not general; classical CSP solvers still
beat them on hand-specified problems. So EV-TRM's edge is **vs LLMs** on
constraint reasoning, plus the **calibrated verification** none of the above
provide together.

## PoC result — 4×4 Sudoku, from scratch (106K–417K params, GPU, ~minutes)

| metric | result |
| ------ | ------ |
| exact-solve (all blanks correct) | **95.8 %** |
| energy head flags its OWN errors (AUC) | **0.81** |
| selective prediction — abstain top-10 % energy | 95.8 % → **97.4 %** |
| selective prediction — abstain top-20 % energy | → **98.3 %** |

Reproduce: `python -m nous.evtrm --side 4 --givens 8 --epochs 90`
(needs ~90 epochs; solve-rate is ~0 until it breaks the plateau around epoch 25,
then climbs to ~96 %).

## What it demonstrates
- The recursive mechanism **solves** at solvable scale (not just on paper).
- The energy head is **calibrated** — predicts when the reasoner is wrong → safe
  abstention and an honest trust signal for an LLM controller. (This is also why
  a *standalone* verifier was inert in earlier experiments: the verifier must be
  trained jointly with the generator, which the energy head is.)

## Honest scope
- 4×4 is a **mechanism PoC**, not SOTA. AUC 0.81 is good, not great.
- 9×9-hard is the real target and needs the full TRM budget.

## Runbook — 9×9 scale-up (persistent GPU, not synchronous notebook)
```
python -m nous.evtrm --side 9 --givens 35 --train-n 20000 --test-n 2000 \
       --epochs 2000 --d 256
```
- Expect solve-rate ~0 for many epochs, then a plateau-break (as at 4×4).
- ~hours–days on 1 GPU; checkpoint/resume recommended (the synchronous-cell
  Colab path used for the PoC cannot host a run this long).
- For the strongest result, adopt the official TRM training recipe (deep
  supervision schedule, 1k-example + heavy augmentation, EMA) on the 9×9 grid
  and attach the same energy head.
```
```
