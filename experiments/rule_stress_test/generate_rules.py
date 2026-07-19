"""
Synthetic rule sets with a KNOWN, programmatically-verified answer key, for
testing whether plain LLM reasoning already does local-to-global contradiction
detection before building NOUS-S (sheaf + energy + active probing).

Domain: hospital nurse scheduling (same flavor as the earlier CSP experiments
in this repo). Every case has a "core" -- a small set of rules that together
prove one of four ground-truth statuses -- padded with independently-generated
DISTRACTOR rules (different nurses/shifts/dates, never touching the core) up
to the target rule count. Rule order is shuffled last; the answer key's rule
IDs are always the POST-shuffle numbering.

Case types:
  A  buried inconsistency  -- target shift has zero feasible certified nurse;
     proving it chains the shift's requirement through every candidate's
     exclusion reason (leave / day-shift-only cert / day-off+fatigue).
  B  underspecified        -- one candidate's status hinges on a fact the
     rules never state (whether a recertification actually completed).
     Correct answer is NOT "inconsistent" -- it's "cannot decide without X".
  C  consistent but suspicious -- same surface tension as A, but a resolving
     rule (float-pool nurse, or an off-by-one date that doesn't actually
     overlap the shift) makes a valid assignment exist.
  D  multiple obstructions -- two independent target shifts, each with its
     own non-overlapping buried-inconsistency core.

Honest scope: "hop count" below is the SIZE of the constructed minimal
obstruction set, not a formally verified graph-distance metric -- it's a
reasonable proxy (each hop = one additional rule that must be combined),
not a proof of minimum chain length.
"""

import argparse
import json
import random

CERTS = ["ICU", "NICU", "OR"]
UNITS = {"ICU": "ICU", "NICU": "Neonatal ICU", "OR": "Operating Room"}
WEEKDAY_DATES = ["Mon Mar 2", "Tue Mar 3", "Wed Mar 4", "Thu Mar 5", "Fri Mar 6"]
WEEKEND_DATES = ["Sat Mar 7", "Sun Mar 8"]

CORE_NAMES = [f"Nurse {c}" for c in "ABCDEFGHIJKL"]
DISTRACTOR_NAMES = [f"Nurse {c}{d}" for c in "PQRSTUVWXYZ" for d in "123456789"]


def _base_cert(nurse, cert):
    return (f"base_cert_{nurse}_{cert}", f"{nurse} is certified in {cert}.")


def _req(shift_id, unit, date, cert):
    return (f"req_{shift_id}", f"The {unit} night shift on {date} requires a nurse certified in {cert}.")


def _weekday_fact(shift_id, date):
    return (f"dow_{shift_id}", f"{date} is a weekday.")


def exclusion_dayshift_only(nurse, cert):
    """1 extra rule: candidate can't cover a NIGHT shift at all."""
    return [(f"exc_{nurse}_dayonly", f"{nurse} is certified in {cert} only for day shifts, per bylaw 4.2.")]


def exclusion_leave(nurse, date):
    """1 extra rule: candidate is on leave covering the date."""
    start, end = "Mar 1", "Mar 9"
    return [(f"exc_{nurse}_leave", f"{nurse} is on approved leave from {start} to {end}.")]


def exclusion_dayoff_fatigue(nurse, date, shift_id, shared_fatigue_rule):
    """3 extra rules (2 own + 1 shared per shift): requested date off AND fatigue-barred."""
    rules = [
        (f"exc_{nurse}_dayoff", f"{nurse} requested {date} off."),
        (f"exc_{nurse}_nights", f"{nurse} worked night shifts on the three consecutive nights before {date}."),
    ]
    if not shared_fatigue_rule["used"]:
        rules.append((f"fatigue_rule_{shift_id}", "Any nurse who works three consecutive night shifts must be off the following night shift."))
        shared_fatigue_rule["used"] = True
    return rules


EXCLUSION_CYCLE = [exclusion_dayshift_only, exclusion_leave, exclusion_dayoff_fatigue]


def build_inconsistent_core(rng, shift_id, n_candidates, tag_prefix=""):
    """One target shift with n_candidates certified nurses, ALL excluded."""
    cert = rng.choice(CERTS)
    unit = UNITS[cert]
    date = rng.choice(WEEKDAY_DATES)
    core = [_req(shift_id, unit, date, cert), _weekday_fact(shift_id, date)]
    shared_fatigue = {"used": False}
    names = rng.sample(CORE_NAMES, n_candidates)
    prefixed_names = [f"{tag_prefix}{n}" for n in names]
    for i, nurse in enumerate(prefixed_names):
        core.append(_base_cert(nurse, cert))
        exc_fn = EXCLUSION_CYCLE[i % len(EXCLUSION_CYCLE)]
        if exc_fn is exclusion_dayshift_only:
            extra = exc_fn(nurse, cert)
        elif exc_fn is exclusion_leave:
            extra = exc_fn(nurse, date)
        else:
            extra = exc_fn(nurse, date, shift_id, shared_fatigue)
        core.extend(extra)
    return core, {"cert": cert, "unit": unit, "date": date, "candidates": prefixed_names}


def make_distractor_block(rng, idx, reserved_certs):
    """Independently self-consistent rules about entities never referenced
    by any core -- some look superficially tense but resolve locally.

    reserved_certs are the cert(s) a case's target shift(s) actually need.
    Distractors must never claim one of those certs UNCONDITIONALLY: with
    dozens of distractors and only 3 certs total, an unattached "X is
    certified in <reserved cert>" is exactly a real, valid, uncounted
    candidate -- it would silently flip an intended "inconsistent" case to
    genuinely consistent (found the hard way: a solver correctly used one
    such stray distractor to resolve a case meant to be unsolvable)."""
    nurse = DISTRACTOR_NAMES[idx % len(DISTRACTOR_NAMES)]
    safe_certs = [c for c in CERTS if c not in reserved_certs] or CERTS
    cert = rng.choice(safe_certs)
    date = rng.choice(WEEKDAY_DATES + WEEKEND_DATES)
    templates = [
        f"{nurse} is certified in {cert}.",
        f"{nurse} prefers day shifts but has no restriction on record.",
        f"{nurse} completed annual safety training on {date}.",
        f"{nurse} is scheduled for the {UNITS[cert]} day shift on {date}.",
        f"{nurse} is not certified in any specialty unit.",
        f"{nurse} requested {date} off and has no shift assigned that day.",
    ]
    return [(f"distractor_{idx}", rng.choice(templates))]


def pad_with_distractors(rng, rules, target_size, reserved_certs):
    idx = 0
    while len(rules) < target_size:
        rules.extend(make_distractor_block(rng, idx, reserved_certs))
        idx += 1
    return rules


def shuffle_and_number(rng, rules):
    """numbered maps tag -> list of final IDs (usually length 1; a tag can
    legitimately repeat, e.g. the same weekday fact restated for two shifts)."""
    order = list(range(len(rules)))
    rng.shuffle(order)
    numbered = {}
    final = []
    for new_id, old_idx in enumerate(order, start=1):
        tag, text = rules[old_idx]
        numbered.setdefault(tag, []).append(new_id)
        final.append((new_id, text))
    final.sort(key=lambda r: r[0])
    return final, numbered


def minimal_ids(numbered, core):
    ids = set()
    for tag, _ in core:
        ids.update(numbered[tag])
    return sorted(ids)


def case_a(rng, size, hops_target=5):
    n_candidates = 2 if hops_target <= 5 else 3
    core, meta = build_inconsistent_core(rng, "S1", n_candidates)
    rules = pad_with_distractors(rng, list(core), size, {meta["cert"]})
    final, numbered = shuffle_and_number(rng, rules)
    minimal = minimal_ids(numbered, core)
    return {
        "case_type": "A", "status": "inconsistent",
        "rules": final, "minimal_obstruction_rules": minimal,
        "best_probe": f"Is there an ICU-certified float or agency nurse available for the {meta['unit']} "
                       f"night shift on {meta['date']}, or can an administrative override waive the "
                       f"fatigue-rest requirement given no certified alternative exists?",
        "notes": meta,
    }


def case_b(rng, size):
    cert = rng.choice(CERTS)
    unit = UNITS[cert]
    date = rng.choice(WEEKDAY_DATES)
    nurse = rng.choice(CORE_NAMES)
    core = [
        _req("S1", unit, date, cert),
        _weekday_fact("S1", date),
        ("ambiguous_cert", f"{nurse}'s {cert} certification requires re-testing every 2 years; "
                            f"{nurse}'s last test was two years ago and the retest has not been confirmed."),
    ]
    rules = pad_with_distractors(rng, list(core), size, {cert})
    final, numbered = shuffle_and_number(rng, rules)
    minimal = minimal_ids(numbered, core)
    return {
        "case_type": "B", "status": "underspecified",
        "rules": final, "minimal_obstruction_rules": minimal,
        "best_probe": f"Has {nurse}'s {cert} recertification test result come back, and is the certification "
                       f"currently active for {date}?",
        "notes": {"cert": cert, "date": date, "nurse": nurse},
    }


def case_c(rng, size):
    cert = rng.choice(CERTS)
    unit = UNITS[cert]
    date = rng.choice(WEEKDAY_DATES)
    n1, n2 = rng.sample(CORE_NAMES, 2)
    core = [
        _req("S1", unit, date, cert),
        _weekday_fact("S1", date),
        _base_cert(n1, cert),
        (f"exc_{n1}_leave", f"{n1} is on approved leave from Mar 10 to Mar 14."),   # looks alarming, doesn't cover `date`
        _base_cert(n2, cert),
        ("float_pool", f"{n2} is a float-pool nurse certified in {cert}, available any day, any shift, "
                        f"and is not assigned elsewhere on {date}."),
    ]
    rules = pad_with_distractors(rng, list(core), size, {cert})
    final, numbered = shuffle_and_number(rng, rules)
    return {
        "case_type": "C", "status": "consistent",
        "rules": final, "minimal_obstruction_rules": [],
        "best_probe": "",
        "notes": {"cert": cert, "date": date, "resolving_nurse": n2, "also_certified_but_leave": n1},
    }


def case_d(rng, size, hops_target=5):
    core1, meta1 = build_inconsistent_core(rng, "S1", 2, tag_prefix="X")
    core2, meta2 = build_inconsistent_core(rng, "S2", 2, tag_prefix="Y")
    rules = pad_with_distractors(rng, list(core1) + list(core2), size, {meta1["cert"], meta2["cert"]})
    final, numbered = shuffle_and_number(rng, rules)
    minimal = minimal_ids(numbered, core1 + core2)
    return {
        "case_type": "D", "status": "inconsistent",
        "rules": final, "minimal_obstruction_rules": minimal,
        "best_probe": f"For both {meta1['unit']} on {meta1['date']} and {meta2['unit']} on {meta2['date']}: "
                       f"is any float/agency-certified nurse available, or can an override apply?",
        "notes": {"core1": meta1, "core2": meta2},
    }


CASE_BUILDERS = {"A": case_a, "B": case_b, "C": case_c, "D": case_d}


def generate_case(case_type, size, seed, shuffle_idx=0):
    rng = random.Random(seed * 1000 + shuffle_idx)
    hops = 5 if size <= 60 else 7
    if case_type in ("A", "D"):
        return CASE_BUILDERS[case_type](rng, size, hops_target=hops)
    return CASE_BUILDERS[case_type](rng, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[60, 100, 150])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--shuffles", type=int, default=2)
    ap.add_argument("--out", default="answer_key.json")
    args = ap.parse_args()

    grid = []
    for case_type in "ABCD":
        for size in args.sizes:
            for seed in range(args.seeds):
                for sh in range(args.shuffles):
                    case = generate_case(case_type, size, seed, sh)
                    grid.append({"case_type": case_type, "size": size, "seed": seed, "shuffle": sh, **case})
    with open(args.out, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"wrote {len(grid)} cases to {args.out}")


if __name__ == "__main__":
    main()
