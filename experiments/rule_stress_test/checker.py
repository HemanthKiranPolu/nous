"""
Deterministic feasibility checker for variant C ("LLM extracts facts -> code
decides"). Takes the structured JSON an LLM extracts from a rule list (see
prompts/template_extract.txt for the schema) and computes status +
minimal_obstruction_rules + a probe by pure logic -- no LLM judgment involved
in the decision, only in the extraction.

Honest scope: the checker never needs a "day X is a weekday" fact (a
day-shift-only restriction excludes a night shift regardless of weekday), so
its minimal-obstruction output never includes that rule even though the
generator's ground key does (it was added as connective narrative, not a
logical necessity). This caps variant C's recall at (k-1)/k where k = the
true minimal-set size whenever a day-shift-only exclusion is in play -- a
known, disclosed artifact, not a checker bug.
"""

def day_num(date_str):
    """'Thu Mar 5' or 'Mar 5' -> 5. All dates in this domain are in March."""
    return int(date_str.strip().split()[-1])


def in_range(date_str, start_str, end_str):
    return day_num(start_str) <= day_num(date_str) <= day_num(end_str)


def _hard_exclusion(nurse, cert, date, day_only, leaves, nights, fatigue):
    """Rule ids excluding `nurse` from `cert` work on `date`, or None if not excluded
    by any DEFINITE (non-ambiguous) rule."""
    restr = next((r for r in day_only if r["nurse"] == nurse and r["cert"] == cert), None)
    if restr:
        return [restr["rule"]]
    leave = next((l for l in leaves if l["nurse"] == nurse and in_range(date, l["start"], l["end"])), None)
    if leave:
        return [leave["rule"]]
    night = next((n for n in nights if n["nurse"] == nurse and day_num(n["before_date"]) == day_num(date)), None)
    if night and fatigue:
        return [night["rule"], fatigue[0]["rule"]]
    return None


def check(facts):
    reqs = facts.get("requirements", [])
    certs = facts.get("certifications", [])
    day_only = facts.get("day_shift_only_restrictions", [])
    leaves = facts.get("leaves", [])
    nights = facts.get("consecutive_nights", [])
    fatigue = facts.get("fatigue_rule_present", [])
    floats = facts.get("float_pool", [])
    ambiguous = facts.get("ambiguous_certifications", [])

    all_obstructions = []
    worst_status = "consistent"
    probes = []

    for req in reqs:
        cert, date = req["cert_required"], req["date"]
        req_rule = req["rule"]
        outcomes = []                                        # (nurse, True/False/None, [rule ids])

        for c in certs:
            if c["cert"] != cert:
                continue
            reasons = [c["rule"]]
            excl = _hard_exclusion(c["nurse"], cert, date, day_only, leaves, nights, fatigue)
            outcomes.append((c["nurse"], False, reasons + excl) if excl else (c["nurse"], True, reasons))

        for a in ambiguous:
            if a["cert"] != cert:
                continue
            reasons = [a["rule"]]
            excl = _hard_exclusion(a["nurse"], cert, date, day_only, leaves, nights, fatigue)
            outcomes.append((a["nurse"], False, reasons + excl) if excl else (a["nurse"], None, reasons))

        for fp in floats:
            if fp["cert"] == cert:
                outcomes.append((fp["nurse"], True, [fp["rule"]]))

        if any(ok is True for _, ok, _ in outcomes):
            continue                                          # this requirement is satisfiable, no obstruction here
        obstruction_rules = {req_rule}
        for _, _, rs in outcomes:
            obstruction_rules.update(rs)
        if any(ok is None for _, ok, _ in outcomes):
            worst_status = "underspecified" if worst_status == "consistent" else worst_status
            probes.append(f"Is the ambiguous certification for one of the candidates for {cert} on {date} confirmed?")
        else:
            worst_status = "inconsistent"                     # inconsistent trumps underspecified if both occur
            probes.append(f"Is any other {cert}-certified nurse or override available for {date}?")
        all_obstructions.append(sorted(obstruction_rules))

    if worst_status == "consistent":
        return {"status": "consistent", "minimal_obstruction_rules": [], "best_probe": "", "confidence": 0.9}
    merged = sorted(set().union(*all_obstructions)) if all_obstructions else []
    return {"status": worst_status, "minimal_obstruction_rules": merged,
            "best_probe": probes[0] if probes else "", "confidence": 0.9}


def _demo():
    # Case A shape: one requirement, two candidates, both excluded.
    facts = {
        "requirements": [{"rule": 10, "cert_required": "NICU", "date": "Thu Mar 5"}],
        "certifications": [{"rule": 36, "nurse": "Nurse A", "cert": "NICU"},
                            {"rule": 47, "nurse": "Nurse E", "cert": "NICU"}],
        "day_shift_only_restrictions": [{"rule": 55, "nurse": "Nurse A", "cert": "NICU"}],
        "leaves": [{"rule": 25, "nurse": "Nurse E", "start": "Mar 1", "end": "Mar 9"}],
        "consecutive_nights": [], "fatigue_rule_present": [], "float_pool": [], "ambiguous_certifications": [],
    }
    r = check(facts)
    assert r["status"] == "inconsistent", r
    assert set(r["minimal_obstruction_rules"]) == {10, 36, 47, 55, 25}, r

    # Case C shape: float pool resolves it -> consistent.
    facts["float_pool"] = [{"rule": 80, "nurse": "Nurse Z", "cert": "NICU"}]
    r = check(facts)
    assert r["status"] == "consistent" and r["minimal_obstruction_rules"] == [], r

    # Case B shape: ambiguous cert, no other candidate -> underspecified.
    facts2 = {
        "requirements": [{"rule": 10, "cert_required": "NICU", "date": "Thu Mar 5"}],
        "certifications": [], "day_shift_only_restrictions": [], "leaves": [],
        "consecutive_nights": [], "fatigue_rule_present": [], "float_pool": [],
        "ambiguous_certifications": [{"rule": 43, "nurse": "Nurse A", "cert": "NICU"}],
    }
    r2 = check(facts2)
    assert r2["status"] == "underspecified", r2

    print("checker.py self-check OK")


if __name__ == "__main__":
    _demo()
