"""
NOUS Agent — a self-growing, fault-tolerant knowledge system on a FROZEN LLM.

The "unfrozen LLM", done safely: the base model's weights never change. Knowledge
grows in an external **librarian memory** — evidence-gated consolidation + editable
semantic ids + a defer gate — and conditions a frozen Qwen (via Ollama) at answer
time (retrieval-augmented). No weight updates → no catastrophic forgetting, no
model-collapse loop, no poisoning-by-single-observation.

Two properties the user asked for, honestly scoped:
  • self-GROW  — real. Memory persists on disk and accretes concepts as you ingest.
  • self-HEAL  — real *fault-tolerance*, NOT magic. Each component has a health
                 check + fallback; on a KNOWN failure (embedder/LLM down, corrupt
                 store) the supervisor retries, degrades gracefully, and keeps
                 running. Autonomous repair of arbitrary novel breakage is beyond
                 the state of the art — this does not claim it.

CLI:
  python -m nous.agent ingest "<text>"      add evidence (consolidates on corroboration)
  python -m nous.agent ask "<question>"     retrieve memory + answer with frozen Qwen
  python -m nous.agent status               health of every component
  python -m nous.agent heal                 run health checks + auto-recover
  python -m nous.agent forget <id>          delete a consolidated concept (editable memory)
"""

import hashlib
import json
import math
import pathlib
import sys
import time
import urllib.request

MEM_PATH = pathlib.Path("results/agent_memory.json")
OLLAMA = "http://localhost:11434"
GEN_MODEL = "qwen2.5:14b-instruct"               # frozen; never trained
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K, CONSISTENCY, RADIUS, TAU = 2, 0.6, 0.9, 0.55  # evidence gate + defer thresholds


def log(kind, msg):
    print(f"[{time.strftime('%H:%M:%S')}] {kind:7s} {msg}", file=sys.stderr)


# ── Embedder: real model, with a deterministic hashing FALLBACK ──────────────
class Embedder:
    def __init__(self):
        self.model = self.tok = None
        self.dim = 384
        self.mode = "cold"

    def _load(self):
        import torch                              # noqa: local import so the CLI starts without torch
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL).eval()
        self.torch, self.mode = torch, "model"

    def healthy(self):
        return self.mode in ("model", "hash")

    def recover(self):
        try:
            self._load(); log("HEAL", "embedder → model")
        except Exception as e:                    # graceful degrade, never crash the agent
            self.mode = "hash"; log("HEAL", f"embedder → hash fallback ({type(e).__name__})")

    def _hash_vec(self, text):
        v = [0.0] * self.dim
        for tok in text.lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, text):
        if self.mode == "cold":
            self.recover()
        if self.mode == "model":
            try:
                import torch
                e = self.tok(text[:2000], return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    h = self.model(**e).last_hidden_state
                m = e["attention_mask"].unsqueeze(-1).float()
                v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)[0]
                return v.tolist()
            except Exception as e:
                log("HEAL", f"embed failed → hash fallback ({type(e).__name__})")
                self.mode = "hash"
        return self._hash_vec(text)               # fallback path


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _entropy(centers, e):
    xs = [-_dist2(c, e) / 0.3 for c in centers]
    m = max(xs); ex = [math.exp(x - m) for x in xs]; s = sum(ex)
    p = [x / s for x in ex]
    return -sum(x * math.log(x + 1e-12) for x in p)


# ── Librarian memory: persistent, evidence-gated, editable, defers ambiguity ──
class Memory:
    def __init__(self, emb):
        self.emb = emb
        self.ids, self.prov, self.next_id = [], [], 1
        self.load()

    def load(self):
        try:
            if MEM_PATH.exists():
                d = json.loads(MEM_PATH.read_text())
                self.ids, self.next_id = d["ids"], d.get("next_id", len(d["ids"]) + 1)
                self.prov = [{**p, "sources": set(p["sources"])} for p in d.get("prov", [])]
                log("MEM", f"loaded {len(self.ids)} concepts, {len(self.prov)} provisional")
        except Exception as e:                    # corrupt store → back up, start clean (self-heal)
            bak = MEM_PATH.with_suffix(".corrupt.json")
            if MEM_PATH.exists():
                MEM_PATH.rename(bak)
            self.ids, self.prov, self.next_id = [], [], 1
            log("HEAL", f"memory corrupt → backed up to {bak.name}, fresh store ({type(e).__name__})")

    def save(self):
        MEM_PATH.parent.mkdir(exist_ok=True)
        prov = [{**p, "sources": sorted(p["sources"])} for p in self.prov]
        MEM_PATH.write_text(json.dumps({"ids": self.ids, "prov": prov, "next_id": self.next_id}, indent=1))

    def _nearest(self, items, e):
        if not items:
            return None, math.inf
        d = [_dist2(it["center"], e) for it in items]
        j = min(range(len(d)), key=lambda i: d[i]); return j, d[j] ** 0.5

    def ingest(self, text, source="user"):
        e = self.emb.embed(text)
        if len(self.ids) >= 2:                     # defer: ambiguous blend of known concepts
            C = [it["center"] for it in self.ids]; _, dist = self._nearest(self.ids, e)
            if dist < RADIUS and _entropy(C, e) > TAU:
                return {"action": "deferred", "reason": "ambiguous — needs clearer evidence"}
        j, dist = self._nearest(self.ids, e)
        if j is not None and dist < RADIUS:        # already known → corroborate (evidence++)
            self.ids[j]["evidence"] += 1
            self.ids[j]["sources"] = sorted(set(self.ids[j]["sources"] + [source]))
            self.save(); return {"action": "reinforced", "id": self.ids[j]["id"]}
        pj, pd = self._nearest(self.prov, e)       # provisional evidence
        if pj is not None and pd < RADIUS:
            p = self.prov[pj]; p["hits"] += 1; p["texts"].append(text); p["sources"].add(source)
            p["center"] = [(a * (p["hits"] - 1) + b) / p["hits"] for a, b in zip(p["center"], e)]
        else:
            self.prov.append({"center": e, "hits": 1, "texts": [text], "sources": {source}}); pj = len(self.prov) - 1
        p = self.prov[pj]
        # consolidate on K repeated observations (one-offs never become memory).
        # For UNTRUSTED sources (internet) also require ≥2 distinct sources — a
        # single source can't cement a fact alone (anti-poisoning knob).
        trusted = all(not s.startswith("web") for s in p["sources"])
        if p["hits"] >= K and (trusted or len(p["sources"]) >= 2):
            cid = self.next_id; self.next_id += 1
            self.ids.append({"id": cid, "center": p["center"], "text": p["texts"][-1],
                             "evidence": p["hits"], "sources": sorted(p["sources"])})
            self.prov.pop(pj); self.save()
            return {"action": "consolidated", "id": cid}
        self.save()
        return {"action": "provisional", "hits": p["hits"], "need": K}

    def retrieve(self, query, k=4):
        e = self.emb.embed(query)
        ranked = sorted(self.ids, key=lambda it: _dist2(it["center"], e))
        return ranked[:k]

    def forget(self, cid):
        n = len(self.ids); self.ids = [it for it in self.ids if it["id"] != cid]; self.save()
        return n != len(self.ids)


# ── Generator: frozen Qwen via Ollama, with retry + offline degrade ──────────
class Generator:
    def reachable(self):
        try:
            urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3); return True
        except Exception:
            return False

    def generate(self, prompt, retries=2):
        body = json.dumps({"model": GEN_MODEL, "prompt": prompt, "stream": False}).encode()
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(f"{OLLAMA}/api/generate", body,
                                             {"Content-Type": "application/json"})
                return json.loads(urllib.request.urlopen(req, timeout=180).read())["response"]
            except Exception as e:
                log("HEAL", f"generate attempt {attempt+1} failed ({type(e).__name__}); retrying")
                time.sleep(1.5)
        return None                               # caller degrades to memory-only


# ── Supervisor: health + recovery, and the query flow ────────────────────────
class Agent:
    def __init__(self):
        self.emb = Embedder(); self.mem = Memory(self.emb); self.gen = Generator()

    def status(self):
        return {"embedder": self.emb.mode, "ollama": "up" if self.gen.reachable() else "down",
                "concepts": len(self.mem.ids), "provisional": len(self.mem.prov)}

    def heal(self):
        acted = []
        if not self.emb.healthy():
            self.emb.recover(); acted.append("embedder")
        if not self.gen.reachable():
            log("HEAL", "ollama unreachable — start it with `ollama serve`; agent stays up in degraded (memory-only) mode")
            acted.append("ollama(degraded)")
        return acted or ["all healthy"]

    def ask(self, query):
        facts = self.mem.retrieve(query)
        ctx = "\n".join(f"- {f['text']} (evidence×{f['evidence']}, sources: {', '.join(f['sources'])})" for f in facts)
        if not self.gen.reachable():              # degrade, don't fail
            return f"[LLM offline — memory only]\nRelevant knowledge:\n{ctx or '(memory empty)'}"
        prompt = (f"Use only the corroborated knowledge below; if it does not cover the "
                  f"question, say you don't know.\nKnowledge:\n{ctx or '(none)'}\n\nQ: {query}\nA:")
        out = self.gen.generate(prompt)
        return out if out is not None else f"[LLM unreachable after retries — memory only]\n{ctx}"


def main(argv):
    if not argv:
        print(__doc__); return
    a = Agent(); cmd = argv[0]
    if cmd == "ingest":
        print(a.mem.ingest(" ".join(argv[1:]), source=f"cli@{time.strftime('%Y-%m-%d')}"))
    elif cmd == "ask":
        print(a.ask(" ".join(argv[1:])))
    elif cmd == "status":
        print(json.dumps(a.status(), indent=1))
    elif cmd == "heal":
        print("healed:", a.heal(), "|", json.dumps(a.status()))
    elif cmd == "forget":
        print("forgotten" if a.mem.forget(int(argv[1])) else "no such id")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
