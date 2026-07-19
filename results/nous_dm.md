# NOUS-DM — Dimension-Morphogenic growth PoC

Tests whether growing a new hidden dimension **targeted at the residual**
(Cascade-Correlation, Fahlman & Lebiere 1990: fit a candidate unit to maximize
`|corr(unit(x), residual)|`) beats **untargeted** growth (DEN/Progressive-Nets
style: add a random unit, retrain on task loss) after a representational
shift. Source: `nous/nous_dm.py`.

## Why this direction (and what it drops)
Prior NOUS-F / frustration proposal reduced to mechanisms already covered
elsewhere in this repo (energy-as-confidence, CSP energy relaxation — the
latter already lost to DSATUR/min-conflicts/SA, see project memory). The one
genuinely untested claim was "grow a new *dimension*, not just a new basin,
when the current representation can't explain a pattern." Prior art already
does representation growth (Cascade-Correlation 1990, DEN 2018, Progressive
Nets 2016) — so the only defensible new question is whether *residual-targeted*
growth beats *untargeted* growth, not whether growing beats not growing.

## Task
Raw input = 4 binary factors (color, size, shape, material), always fully
present as one-hot pairs. Label = XOR of the "active" factors. Stage 1 uses
{color, size}; stage 2 adds shape; stage 3 adds material. A model with only 2
hidden units can solve stage 1 but structurally cannot represent the added
XOR terms — it needs a new hidden dimension reading the newly-relevant factor.

Three policies, same "loss plateaued while still bad" growth trigger:
- **fixed** — never grows.
- **generic** — grows a random unit, retrains it via normal backprop (DEN-like).
- **targeted (NOUS-DM)** — fits a candidate unit to the residual, only grows if
  `|corr| > tau` (with weight decay during candidate-fitting — unconstrained
  correlation maximization drives `||w||` into tanh saturation and kills the
  unit's gradient once grown; this was found and fixed during the PoC).

## Result (15 seeds, mean)

| policy | final acc (after 3 shifts) | epochs to recover (≥95% acc) | dims grown | false growth (no-shift control) |
| --- | ---: | ---: | ---: | ---: |
| fixed | 64.4% | never | 0.0 | — |
| generic (DEN-like) | 96.8% | 155.4 | 2.9 | 0.7 / 3 stages |
| **NOUS-DM (targeted)** | 94.8% | **119.9** | 2.8 | 0.9 / 3 stages |

## Honest read
- Growing beats not growing, trivially (fixed collapses to 64%). Not the
  interesting finding.
- **Targeted growth recovers ~23% faster** than untargeted growth after a real
  shift (120 vs 155 epochs) with a comparable/slightly smaller final model —
  the one result that supports the "residual-targeted growth" idea over
  generic capacity-adding.
- Final accuracy is a wash (94.8% vs 96.8%, well within seed noise).
- **Calibration did NOT improve**: false-growth rate under the no-shift
  control is the same or slightly worse for targeted (0.9) vs generic (0.7).
  The shared "loss plateaued above threshold" trigger — not the
  targeting — is what causes false growth: small-XOR training genuinely
  plateaus for a while before escaping on its own, and the residual at that
  point is real, so a residual-correlated candidate finds something to grow
  on even when growth wasn't necessary. Fixing this needs a convergence
  check (e.g. verify the current network truly can't do better with more
  training/restarts) before growing at all — not attempted here, out of
  scope for a PoC.

## Verdict
There is a real, modest effect (faster recovery, not higher ceiling or better
calibration) on a toy task. Not evidence for the "concept formation" framing
in the original proposal — it's Cascade-Correlation with a name change, and
the interesting failure mode (false growth from transient plateaus, not
genuine incapacity) is a bigger problem than the growth-targeting question.
Do not build NOUS-DM into a paper direction on this result alone; if pursued
further, fix the growth trigger first (it's the weaker link, not the
targeting), then retest on a task where "false growth" has a real cost.

Reproduce: `python -m nous.nous_dm --seeds 15`

---

## v2 — Growth TRIGGER, not growth target (2026-07-04)

v1's finding reframed the question: the weak point isn't *what* gets grown,
it's *when*. Tested the falsifiable claim: **"NOUS-DM grows only when
representation is insufficient, not when optimization is temporarily slow."**

Added a **patience probe**: on hitting the plateau trigger, pause, train 30
more epochs at a lower LR (0.015) on a held-out validation split; if val loss
improves, don't grow (the plateau was ordinary optimization difficulty);
if it still stalls, revert the probe training and grow. Five variants
(`nous/nous_dm.py`, `VARIANTS`): DEN/NOUS-DM × {plateau, probe} trigger, plus
NOUS-DM + probe + prune (removes grown dims whose removal doesn't hurt val
loss). Added two new metrics: growth precision (useful grows / total grows,
"useful" = final val loss beats the val loss recorded right before that grow)
and forgetting (accuracy on earlier stages' held-out data, re-measured after
the schedule finishes).

### Result (15 seeds, mean)

| variant | final acc | recover epochs | grows | precision | false grows (no-shift) | earlier-stage acc (forgetting) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DEN + plateau (v1 baseline) | 96.8% | 155.4 | 2.9 | 100.0% | 0.7 | 52.0% |
| NOUS-DM + plateau (v1 result) | 94.8% | **119.9** | 2.8 | 100.0% | 0.9 | 49.9% |
| DEN + probe | 95.0% | 153.8 | 2.6 | 89.7% | 0.6 | 52.4% |
| **NOUS-DM + probe (the new claim)** | 94.2% | 190.7 | 2.6 | 97.4% | 0.8 | 49.9% |
| NOUS-DM + probe + prune | 94.2% | 190.7 | 2.8 | 95.2% | 0.8 | 51.0% |

### Checked against the decision rule (continue only if the new trigger beats DEN+plateau on ALL of: recovery, false grows, accuracy, forgetting)

- **Recovery: FAILS.** 190.7 vs 155.4 epochs — probe-gated targeting is
  *slower*, not faster. Stacking two conservative gates (probe + correlation
  threshold) under-triggers real growth needs, trading recall for a precision
  that doesn't materialize where it matters.
- **False grows: FAILS.** 0.8 vs 0.7 — no improvement, within noise. The
  probe does not reliably separate "representation insufficient" from
  "temporarily slow optimization" on this task; a smaller 6-seed pilot run
  showed an improvement (0.3 vs 0.5) that did not hold up at 15 seeds — a
  reminder to run the larger seed count before believing a trigger fix works.
- **Final accuracy: marginal fail.** 94.2% vs 96.8%, ~2.6pp lower — inside
  plausible seed noise but not "similar or better" as stated.
- **Forgetting: no change.** All five variants land at ~50-52% accuracy on
  earlier stages' data — chance level for a binary label. None of these
  growth mechanisms address catastrophic forgetting; growing capacity doesn't
  help when a single shared linear readout combines all hidden units into one
  output — old and new mappings compete for the same weights regardless of
  how many dimensions exist underneath.
- Pruning never fired in the shift condition (dims identical with/without
  prune) — the grown units were all judged locally necessary by the
  loss-based pruning check, so it's untested whether pruning helps; the
  hypothesis just never got exercised on this task.

### Verdict
The probe-based trigger does not clear the bar its own decision rule set —
it makes recovery worse without fixing false growth, precision, or
forgetting. This closes out the "smarter trigger" refinement on this toy
task: two consecutive PoCs (v1: targeting, v2: triggering) have now each
produced one narrow positive (recovery speed under the dumb trigger) and
several failed hypotheses. Diminishing returns on this synthetic task —
further tuning of trigger heuristics here is unlikely to produce a
paper-worthy result. If this direction is revisited, it needs a task where
false growth and forgetting carry a real cost (so a trigger fix has
something to win), not another parameter sweep on this XOR toy.

Reproduce: `python -m nous.nous_dm --seeds 15`
