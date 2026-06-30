# NOUS as a controller for LLM-style generators — controlled study

Two controlled experiments on the SCAN-mini compositional task settle *which*
controller role actually leverages NOUS. Both reuse the proven setup (NOUS
central, factored decode); no new modelling claims.

## 1. Verifier (accept/reject) — NOUS shows NO edge

`nous/verify_compgen.py`: judge `(command, candidate_output)` valid/invalid;
held out = novel verb×count compositions.

- NOUS and a param-matched MLP both get **valid-recall ≈ 0** on novel
  compositions — both **reject every novel valid candidate**. Δ ≈ 0.
- Why: a discriminative verifier has no constructive signal; a valid novel
  composition just looks *unfamiliar*, so it is rejected. NOUS's strength is
  building the right output, not scoring familiarity.
- **Conclusion: a pure NOUS accept/reject verifier is inert** — same lesson as
  the SCAN-latent ablation. Do not build the "NOUS scores LLM output" system.

## 2. Generative controller — NOUS recovers what the LLM misses

`nous/controller_demo.py`: route novel-composition / structured queries to NOUS
(the compositional generator); the LLM-style baseline (transformer) keeps the
rest.

On novel-composition queries (held-out SCAN-mini pairs, 3 seeds):

| system | exact-match |
| ------ | ----------- |
| LLM alone (transformer) | **0 %** |
| LLM + NOUS controller   | **78 %** (Δ **+78 pp**) |

NOUS **constructs** the valid novel output from known parts — exactly its proven
skill (57.5 % vs 12.5 % / 9.2 %, see `README.md`). The controller hands NOUS the
structured sub-problem it generalizes on.

## Honest caveats

- This is the SCAN-mini **toy**, not real SCAN (which needs NOUS-central at
  scale — open research; the latent variant is proven inert).
- Routing here uses the known novel/structured split. **Production needs a
  router** to decide *when* to invoke NOUS — and NOUS is NOT good at that
  detection (experiment 1). So the routing/trigger is itself a design problem;
  candidates: task-type heuristics, LLM self-uncertainty, or schema/grammar tags
  — not a NOUS verifier.
- The win is scoped to structured/compositional sub-queries, not general
  language. NOUS = constraint/compositional generator, LLM = everything else.

## 3. Real-LLM reality check — the controller has no niche yet

`nous/router.py` routes structured queries to NOUS. But the premise requires the
LLM to actually *fail* where NOUS succeeds. Tested directly: a real instruct LLM
(**Qwen2.5-1.5B-Instruct**, Colab A100) few-shot on the **same** held-out
compositions:

| model | held-out compositions |
| ----- | --------------------- |
| Qwen2.5-1.5B-Instruct (real LLM) | **6/6 = 100 %** |
| NOUS (from scratch) | 57.5 % |
| transformer (from scratch) | 9.2 % |
| MLP (from scratch) | 12.5 % |

The toy "compositional gap" exists **only for tiny from-scratch models**. A real
pretrained LLM solves it perfectly — so the router would hand NOUS queries the
LLM already aces, and NOUS (57 %) would *lower* accuracy. There is currently **no
task with (real-LLM-fails ∩ NOUS-competent)**: the toy is trivial for LLMs, and
the tasks where LLMs fail (real SCAN splits, hard reasoning) are exactly where
NOUS is not competent.

## Takeaway

NOUS's controller value is **generative**, not discriminative — but that value
is unrealized in practice because the only domain NOUS handles is one real LLMs
already solve. **The blocker is NOUS competence on hard tasks (open research),
not system engineering.** Do not ship an LLM+NOUS controller yet; it would route
work to NOUS that LLMs do better.
