"""
Router for an LLM + NOUS controller system.

Decides, per query, who answers:
  - structured/compositional sub-query NOUS was trained on  → NOUS (generator)
  - everything else                                          → LLM

This is the missing piece the controller demo hand-waved (it routed by the known
split). The router must decide WITHOUT the ground-truth label. Here the trigger
is a schema/grammar match: if the query parses as the structured grammar NOUS
owns, route to NOUS; otherwise the LLM handles it. (Production alternatives:
LLM self-uncertainty, task-type tags — but NOT a NOUS verifier; that was shown
inert.)

The honest catch this exposes: the router can only send NOUS work NOUS is
competent on. NOUS is competent only on the toy grammar — which a real LLM
already solves. So on this domain the router will hand NOUS queries the LLM
would also get right. A useful controller needs a domain where the LLM FAILS
and NOUS is competent; that intersection does not yet exist (see results/).
"""

import re

# the structured grammar NOUS owns: "<verb> [once|twice|thrice]"
VERBS = {"walk": 0, "look": 1, "run": 2, "jump": 3, "turn": 4, "stay": 5}
COUNTS = {"once": 0, "twice": 1, "thrice": 2}


def parse_structured(query: str):
    """Return (verb_id, count_id) if the query is in NOUS's grammar, else None."""
    toks = re.findall(r"[a-z]+", query.strip().lower())
    if len(toks) == 1 and toks[0] in VERBS:
        return VERBS[toks[0]], 0
    if len(toks) == 2 and toks[0] in VERBS and toks[1] in COUNTS:
        return VERBS[toks[0]], COUNTS[toks[1]]
    return None


def route(query: str) -> str:
    """'nous' if the query is a structured composition NOUS owns, else 'llm'."""
    return "nous" if parse_structured(query) is not None else "llm"


def system_answer(query: str, llm_fn, nous_fn):
    """End-to-end: route, then dispatch to the chosen handler."""
    if route(query) == "nous":
        v, c = parse_structured(query)
        return "nous", nous_fn(v, c)
    return "llm", llm_fn(query)


if __name__ == "__main__":
    for q in ["jump thrice", "stay twice", "what's the capital of France?", "walk"]:
        print(f"{q!r:35} → {route(q)}")
