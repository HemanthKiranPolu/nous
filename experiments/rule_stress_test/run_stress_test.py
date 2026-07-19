"""
Stress test: does plain LLM reasoning already do local-to-global contradiction
detection at scale, before NOUS-S (sheaf + energy + active probing) earns a build?

Runs generate_rules.generate_case() cases through a local LLM (ollama) at
increasing rule-set sizes, scores strictly against the programmatic answer
key, and reports whether accuracy degrades with scale -- the decision rule
from the NOUS-S proposal.

Three variants (--variant): "direct" (LLM judges from the raw rules), "structured"
(LLM judges, but forced through an explicit checklist first), "extract" (LLM only
extracts structured facts; checker.py decides -- no LLM judgment in the decision).

Run (pilot, ~1 min/trial on a 9B local model):
  python run_stress_test.py --sizes 60 100 --seeds 2 --shuffles 2 --variant direct
  python run_stress_test.py --sizes 60 100 --seeds 2 --shuffles 2 --variant structured
  python run_stress_test.py --sizes 60 100 --seeds 2 --shuffles 2 --variant extract

Decision rule (from the proposal, not invented here):
  Only build NOUS-S if accuracy drops below ~85% at 60 rules, or the model
  misses the minimal obstruction / hallucinates at 100, or answers become
  unstable across shuffles at 150. If plain LLM reasoning holds up through
  all three, NOUS-S is not worth building yet.
"""

import argparse
import json
import statistics
import time
import urllib.request

import checker as checker_mod
from generate_rules import generate_case

TEMPLATES = {
    "direct": "prompts/template.txt",              # A: LLM judges directly
    "structured": "prompts/template_structured.txt",  # B: LLM judges, forced checklist first
    "extract": "prompts/template_extract.txt",      # C: LLM extracts facts, checker.py decides
}


def build_prompt(case, variant):
    template = open(TEMPLATES[variant]).read()
    rule_lines = "\n".join(f"{rid}. {text}" for rid, text in case["rules"])
    return template.replace("{rules}", rule_lines)


def extract_json(text):
    """First balanced {...} object in text, string-aware (ignores braces
    inside quoted strings so reasoning preambles/quoted text can't break it)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def call_ollama(prompt, model, host="http://localhost:11434", timeout=420):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                        "options": {"temperature": 0, "num_predict": 8000}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def score(case, parsed):
    truth = set(case["minimal_obstruction_rules"])
    valid_ids = {rid for rid, _ in case["rules"]}
    out = {"status_correct": False, "precision": None, "recall": None,
           "hallucination": False, "parse_ok": parsed is not None}
    if parsed is None:
        return out
    out["status_correct"] = parsed.get("status") == case["status"]
    pred_raw = parsed.get("minimal_obstruction_rules", [])
    pred = {p for p in pred_raw if isinstance(p, int)}
    out["hallucination"] = any((not isinstance(p, int)) or p not in valid_ids for p in pred_raw)
    if truth:
        hit = pred & truth
        out["precision"] = len(hit) / len(pred) if pred else 0.0
        out["recall"] = len(hit) / len(truth)
    else:                                                    # consistent case: correct answer is []
        out["precision"] = 1.0 if not pred else 0.0
        out["recall"] = 1.0 if not pred else 0.0
    return out


def run_trial(case_type, size, seed, shuffle, model, host, variant):
    case = generate_case(case_type, size, seed, shuffle)
    prompt = build_prompt(case, variant)
    t0 = time.time()
    raw = call_ollama(prompt, model, host)
    dt = time.time() - t0
    extracted = extract_json(raw)
    if variant == "extract":
        try:
            decision = checker_mod.check(extracted) if extracted is not None else None
        except (KeyError, TypeError, IndexError):
            decision = None                                   # malformed extraction -> counts as a miss
    else:
        decision = extracted
    s = score(case, decision)
    s.update({"case_type": case_type, "size": size, "seed": seed, "shuffle": shuffle,
              "seconds": round(dt, 1), "raw_response": raw, "parsed": decision,
              "extracted_facts": extracted if variant == "extract" else None,
              "true_status": case["status"]})
    return s


def summarize(rows, group_keys):
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)
    lines = []
    for key, rs in sorted(groups.items()):
        n = len(rs)
        acc = sum(r["status_correct"] for r in rs) / n
        precs = [r["precision"] for r in rs if r["precision"] is not None]
        recs = [r["recall"] for r in rs if r["recall"] is not None]
        prec = statistics.fmean(precs) if precs else float("nan")
        rec = statistics.fmean(recs) if recs else float("nan")
        halluc = sum(r["hallucination"] for r in rs) / n
        parse_fail = sum(not r["parse_ok"] for r in rs) / n
        label = ", ".join(f"{k}={v}" for k, v in zip(group_keys, key))
        lines.append(f"{label:<28} n={n:3d}  status_acc={acc*100:5.1f}%  "
                      f"precision={prec*100:5.1f}%  recall={rec*100:5.1f}%  "
                      f"hallucination={halluc*100:4.1f}%  parse_fail={parse_fail*100:4.1f}%")
    return lines


def stability(rows):
    """For each (case_type, size, seed), does status agree across shuffles?"""
    groups = {}
    for r in rows:
        key = (r["case_type"], r["size"], r["seed"])
        groups.setdefault(key, []).append(r["parsed"].get("status") if r["parsed"] else None)
    stable = sum(len(set(v)) == 1 for v in groups.values())
    return stable / len(groups) if groups else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[60, 100, 150])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--shuffles", type=int, default=2)
    ap.add_argument("--case-types", default="ABCD")
    ap.add_argument("--model", default="hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--variant", choices=list(TEMPLATES), default="direct")
    ap.add_argument("--out", default="results/raw_trials.json")
    args = ap.parse_args()

    rows = []
    total = len(args.case_types) * len(args.sizes) * args.seeds * args.shuffles
    done = 0
    for ct in args.case_types:
        for size in args.sizes:
            for seed in range(args.seeds):
                for sh in range(args.shuffles):
                    r = run_trial(ct, size, seed, sh, args.model, args.host, args.variant)
                    rows.append(r)
                    done += 1
                    print(f"[{done}/{total}] {ct} size={size} seed={seed} sh={sh} "
                          f"-> status={r['parsed'].get('status') if r['parsed'] else 'PARSE_FAIL'} "
                          f"(true={r['true_status']}) {r['seconds']}s")

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=== by size (all case types) ===")
    for line in summarize(rows, ["size"]):
        print(line)
    print("\n=== by case type ===")
    for line in summarize(rows, ["case_type"]):
        print(line)
    print("\n=== by size x case type ===")
    for line in summarize(rows, ["size", "case_type"]):
        print(line)
    print(f"\nshuffle stability (status agrees across all shuffles of same case): "
          f"{stability(rows)*100:.1f}%")
    print(f"\nraw trials -> {args.out}")


if __name__ == "__main__":
    main()
