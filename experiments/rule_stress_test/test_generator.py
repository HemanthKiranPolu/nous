"""
Self-check for generate_rules.py: verifies, structurally (via tags, not text
parsing), that no padded rule set contains an unaccounted-for certified
candidate for the reserved cert(s) -- the exact bug class found by hand
(a stray distractor silently resolving a case meant to be inconsistent/
underspecified). Run: python test_generator.py
"""

import re

import generate_rules as G

UNRESTRICTED_CERT = re.compile(r"^(.+) is certified in (\w+)\.$")


def reserved_certs_of(case):
    notes = case["notes"]
    if "core1" in notes:
        return {notes["core1"]["cert"], notes["core2"]["cert"]}
    return {notes["cert"]}


def known_ok_for(case):
    """Nurses the case intentionally accounts for (excluded candidates, or
    the deliberate resolver in case C, which lists the resolver in notes)."""
    notes = case["notes"]
    if "core1" in notes:
        return set(notes["core1"]["candidates"]) | set(notes["core2"]["candidates"])
    if "candidates" in notes:
        return set(notes["candidates"])
    if "resolving_nurse" in notes:
        return {notes["resolving_nurse"], notes["also_certified_but_leave"]}
    return {notes["nurse"]}                                   # case B


def find_unaccounted_candidates(case):
    """Scans the RENDERED rules (what an LLM actually sees) for any
    unconditional "X is certified in <reserved cert>." sentence whose nurse
    isn't one the case already accounts for -- exactly the bug class found
    by hand (a stray distractor silently resolving a case meant to be
    inconsistent/underspecified)."""
    reserved = reserved_certs_of(case)
    ok = known_ok_for(case)
    bad = []
    for rid, text in case["rules"]:
        m = UNRESTRICTED_CERT.match(text)
        if m and m.group(2) in reserved and m.group(1) not in ok:
            bad.append((rid, text))
    return bad


def main():
    failures = []
    trials = 0
    for case_type in "ABCD":
        for size in [20, 60, 100, 150]:
            for seed in range(15):
                trials += 1
                case = G.generate_case(case_type, size, seed)
                bad = find_unaccounted_candidates(case)
                if bad:
                    failures.append((case_type, size, seed, bad))
    print(f"{trials} trials, {len(failures)} failures")
    for f in failures[:10]:
        print(f)
    assert not failures, f"{len(failures)} cases have an unaccounted-for certified candidate"
    print("OK: no stray distractor ever claims a reserved cert unconditionally.")


if __name__ == "__main__":
    main()
