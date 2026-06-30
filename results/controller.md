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

## Takeaway

NOUS's controller value is **generative** (build the valid structured output),
not **discriminative** (score accept/reject). Build the repair/completion path,
not the verifier path.
