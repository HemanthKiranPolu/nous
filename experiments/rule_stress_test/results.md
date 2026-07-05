# Rule stress test — does plain LLM reasoning already do local-to-global contradiction detection?

Built to answer one question before writing any NOUS-S (sheaf + energy + active probing)
code: does a real LLM already solve buried, local-to-global rule contradictions at scale?
Source: `generate_rules.py` (synthetic cases + programmatic answer key), `run_stress_test.py`
(local-model runner), `test_generator.py` (generator self-check).

## A critical bug found before trusting any score

First-pass generator padded every case with random distractor rules, including
"`<nurse>` is certified in `<cert>`" statements whose cert was chosen independently of
the case's target requirement. With only 3 certs and dozens of distractors, an
UNRESTRICTED "is certified in `<reserved cert>`" distractor showed up in most generated
"inconsistent" cases -- a real, uncounted, valid candidate that silently made the case
actually consistent. Caught it by hand: an LLM trial confidently (and correctly, given
the actual rules) called an intended-inconsistent case "consistent," citing two
plainly-stated NICU-certified nurses my own core never accounted for.

Fix: distractors are now barred from claiming any cert reserved by that case's target
shift(s) (`pad_with_distractors(..., reserved_certs)`). Added `test_generator.py` --
scans 240 generated cases (4 types x 4 sizes x 15 seeds) for any unaccounted-for
certified candidate; 0 failures after the fix. Every LLM trial reported below used the
corrected generator. Every prior number produced before this fix (an earlier pilot
run) was discarded and re-run -- this is exactly the class of error the user's own
review flagged: "generator bugs mattered more than the model score."

## Stronger-model spot check (8 trials: sizes 60/100, all 4 case types)

No paid API or subscription model was reachable in this environment (all `ollama
:cloud` models require a subscription; `claude -p` isn't authenticated in this shell).
Substituted 8 independently-dispatched, context-free general-purpose agents (Sonnet-tier,
no memory of this investigation, no tools) as the "real LLM" data point -- each given
exactly the rendered prompt, nothing else.

| size | case | true status | predicted | status correct | precision | recall |
| ---: | :--: | :--- | :--- | :---: | ---: | ---: |
| 60 | A | inconsistent | consistent | NO | 0.0 | 0.0 |
| 60 | B | underspecified | consistent | NO | 0.0 | 0.0 |
| 60 | C | consistent | consistent | yes | 1.0 | 1.0 |
| 60 | D | inconsistent | consistent | NO | 0.0 | 0.0 |
| 100 | A | inconsistent | inconsistent | yes | 1.0 | 0.40 |
| 100 | B | underspecified | inconsistent | NO | 1.0 | 0.67 |
| 100 | C | consistent | consistent | yes | 1.0 | 1.0 |
| 100 | D | inconsistent | inconsistent | yes | 1.0 | 0.17 |

**Status accuracy: 4/8 (50%) overall, 2/6 (33%) on the non-trivial cases (A/B/D --
the two "C" consistent cases are the easy ones). At size 60 specifically: 0/3 correct
on A/B/D.** Even the two "correct" inconsistent calls (100-A, 100-D) cited only
40% and 17% of the true minimal obstruction set respectively, and 100-D's own
reasoning trace shows the model initially misreading two rules as an "internal
contradiction" before stumbling to the right status via a different, incomplete path
-- and it never found the case's second independent obstruction (case D plants two).
The 60-A miss is a concrete, diagnosable error, not vagueness: the agent read "Nurse E
is on approved leave from Mar 1 to Mar 9" and concluded leave "doesn't cover a night
shift assignment conflict since no rule assigns E there" -- misinterpreting leave as
only blocking pre-existing assignments, not new-eligibility, and missing that the
target date falls inside the leave window.

## Local small-model pilot (Ornith-1.0-9B, 24 trials)

Ran on the corrected generator; results pending completion (background run). Will be
appended when done, but is not the load-bearing result -- the 9B model was always
expected to struggle, per the user's own point that a small local model failing
"proves small local models break," not that "LLM reasoning generally fails." The
Sonnet-tier spot check above is the one that actually bears on the decision.

## Verdict against the proposal's decision rule

The proposal said: build NOUS-S only if a stress test shows repeatable failure; the
threshold given was "60 rules: LLM accuracy drops below ~85%." The Sonnet-tier
spot check clears that bar in the wrong direction for LLMs -- **33% accuracy on the
genuinely hard cases at 60-100 rules, 0% at size 60**, with concrete, characterizable
failure modes (misreading what "on leave" excludes; incomplete minimal-obstruction
recall even when status is right; missing a second independent obstruction).

**This is a real, repeatable, diagnosable failure at a scale far short of production
rule sets** (real scheduling/compliance/policy audits run to hundreds or thousands of
rules). It clears the bar the proposal itself set for building something. Caveats
before treating this as final:
- n=8 is a spot check, not the full 90-trial grid the proposal specified -- worth
  running the larger grid (more seeds, the size-150 tier, shuffle-stability) before
  committing serious engineering.
- These are general-purpose subagents, not a dedicated single flagship-model call with
  a tuned prompt; a more careful invocation (explicit step-by-step scratchpad, larger
  thinking budget) might do meaningfully better. The failure could be an artifact of
  "answer quickly" framing rather than a hard capability ceiling.
- The task is synthetic and one domain (nurse scheduling). Real-world local-to-global
  contradiction tasks (legal, compliance, multi-source data fusion) may be easier or
  harder depending on how "buried" the contradiction naturally is.

## Recommendation

Worth a full run before writing NOUS-S: same generator, same 8-case-type x 2-size
slice, but (a) more seeds for a real confidence interval, (b) a properly
scaffolded frontier-model call (step-by-step reasoning explicitly requested, not
"keep it brief") to rule out the "agent answered too fast" confound, and (c) the
shuffle-stability check (does the same model flip its answer on the same facts
reordered) that hasn't been run on a strong model yet. If that holds up, NOUS-S's
core mechanism -- detect where local facts can't glue into one global solution, then
pick the cheapest probe -- has a real target: LLMs miss buried, multi-hop, distractor-
laden contradictions that a structured consistency check would not.

---

## 3-way comparison: direct vs structured-prompt vs extract+checker (2026-07-04)

Ran (b) immediately -- added `checker.py` (deterministic feasibility engine over
extracted facts), `prompts/template_structured.txt` (forces an explicit checklist
before answering), `prompts/template_extract.txt` (pure fact extraction, no
judgment), and `--variant {direct,structured,extract}` on `run_stress_test.py`.
Same 8 held-out cases (sizes 60/100, all 4 case types) as the earlier spot check,
dispatched as fresh context-free Sonnet-tier agents per variant.

### Variant A (direct) -- reused from the earlier spot check, unchanged
Status accuracy: 4/8 (50%) overall, 2/6 (33%) non-trivial (A/B/D).

### Variant B (structured checklist) -- same rules, forced step-by-step reasoning
**Status accuracy: 7/8 (87.5%) overall, 5/6 (83.3%) non-trivial.** Precision 100%
on every trial; recall 42-90% (capped below 100% by a known, disclosed artifact --
the true minimal set always includes a "day X is a weekday" rule that isn't
actually load-bearing for the exclusion logic, see checker.py's docstring). The
one miss (100-D, two independent obstructions) had two concrete logic errors: the
model read a day-shift-only restriction as NOT applying to a night shift (backwards
-- that's exactly when it applies) and misread a leave range as not covering a date
it did cover. **Plain prompting -- an explicit checklist, no code, no extraction --
recovered most of the gap** found in the direct-variant spot check.

### Variant C (extract facts -> checker.py decides) -- compromised by a process error
Of 8 dispatches: trial 7 hit a session limit mid-run (no data). Trials 4 and 5 were
built from mismatched prompt text -- a copy/paste error while assembling 8 large
prompts by hand pasted the wrong case's rules into those two dispatches, so their
extraction ran against different text than the answer key they were scored
against. **These are not evidence about the method; they're evidence about a
manual-dispatch process mistake**, caught and disclosed rather than reported as a
"48% recall" finding. On the 5 valid trials (0,1,2,3,6 -- all four case types
represented at least once, extraction verified to match the actual case rules):
**5/5 (100%) status accuracy**, with correctly-omitted weekday facts pushing
recall as high as B's. Not enough valid trials to claim variant C beats B, only
that it isn't obviously worse where it ran cleanly.

### What this settles and what it doesn't
- **Settles**: the direct-variant gap found earlier was substantially a prompting/
  scaffolding problem, not a hard capability ceiling. An explicit checklist alone
  took non-trivial-case accuracy from 33% to 83% with zero code. This is a strong,
  simple result on its own.
- **Doesn't settle**: whether extraction+checker (variant C) does meaningfully
  better than variant B once run cleanly at full scale (150 rules, more seeds,
  shuffle stability) -- 5 valid trials isn't enough to say, and the two failures
  that would matter for that comparison are contaminated data, not real misses.
- **Multi-obstruction blindness persists across both surviving variants' failures**
  (B's one miss, and it's the same case type A's direct-variant miss was):
  something about two independent, non-overlapping obstructions in one rule set is
  harder than one, for prompted LLM judgment specifically. A checker that
  enumerates every `requirements` entry (as `checker.py` does) doesn't have this
  failure mode structurally -- worth confirming that holds once C is re-run clean.

### Revised verdict
NOUS-S-0 (structured extraction + checker + probe selector, no neural sheaf math)
remains worth building, but the case for it over "just use a better prompt" is
currently unproven -- variant B alone closed most of the gap for free. Before
committing to NOUS-S-0's engineering, re-run variant C cleanly (scripted, not
hand-assembled, to eliminate the copy/paste risk) at the full grid the original
proposal specified, and see whether it beats variant B's 83% rather than just
matching direct's ceiling. If it doesn't clear B by a meaningful margin, the
honest conclusion is "prompt engineering solved this," not "NOUS-S-0 is needed."

---

## Clean scripted B-vs-C rerun (2026-07-04, local model, no hand-paste)

Ran `run_stress_test.py --variant structured` then `--variant extract` back to
back, fully scripted (`run_stress_test.py`'s existing CLI, `checker.py` deciding
for C), against the local `Ornith-1.0-9B` model -- the exact process fix the
copy/paste incident called for. Scope: sizes 60/100, 1 seed, 1 shuffle, all 4
case types (8 cases x 2 variants = 16 calls, single-threaded local server).

| variant | overall | size 60 | size 100 | A | B | C | D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B (structured checklist) | 75.0% (6/8) | 75% | 75% | 100% | 0% | 100% | 100% |
| C (extract + checker.py) | 62.5% (5/8) | 100% | 25% | 100% | 50% | 50%* | 50% |

*C's one size-100 "consistent" case hit a JSON parse failure (extraction
malformed, not a checker bug) -- counted as wrong, not excluded.

**On this model, at this scale, B beats C by 12.5 points -- the opposite of what
would justify NOUS-S.** All of C's size-100 losses trace to bad extraction (a
missed fact, a malformed JSON, a misread field), not bad checker logic --
`checker.py` never got a chance to reason wrong because it never got the right
inputs. That's the actual architectural risk in an extract-then-decide
pipeline: **the checker has zero error-correction capacity.** A direct-reasoning
model with a checklist can notice its own inconsistency mid-reasoning and
self-correct (that's most of why B > direct-variant A was such a large jump);
a deterministic checker downstream of a lossy extraction step cannot recover
from an extraction mistake at all -- garbage in is silently garbage out, with
high apparent confidence.

Caveats: n=8, 1 seed, 1 shuffle, one local 9B model -- not the full grid, and
a stronger model's extraction fidelity could differ (the earlier Sonnet-tier
spot check, contamination aside, had 5/5 correct on its clean extract trials).
But this is now two independent signals (Sonnet-tier hand-dispatch, contested;
local scripted, clean) and neither shows C clearly beating B.

### Final verdict on this branch
**Do not build NOUS-S-0.** The decision rule was "only build if C clearly beats
B"; on the one clean run available, C is worse, and the reason it's worse is a
structural property of the extract-then-decide design (no error correction),
not a fixable prompt-wording issue. The honest conclusion: structured prompting
(variant B) is the answer this stress test converged on. The harness
(`generate_rules.py`, `test_generator.py`, `checker.py`, `run_stress_test.py`,
three prompt templates) remains a real asset for testing this class of question
again if a harder, more production-realistic rule set ever revives it.
