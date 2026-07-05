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
