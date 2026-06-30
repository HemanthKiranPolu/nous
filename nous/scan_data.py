"""
SCAN dataset loader (Lake & Baroni 2018).

Downloads and parses the official SCAN splits from the brendenlake/SCAN repo,
caching to disk. Each example is a line:

    IN: jump twice OUT: I_JUMP I_JUMP

The interesting generalization split is `addprim_jump`: `jump` is seen only in
isolation at train time, so every composed `jump ...` command at test time is a
known primitive in novel structural contexts — the canonical place seq2seq
transformers collapse.

CLI:
  python -m nous.scan_data --split addprim_jump      # download + print stats
"""

import argparse
import os
import urllib.request

_BASE = "https://raw.githubusercontent.com/brendenlake/SCAN/master/"

# split name → (train suffix, test suffix)
SPLITS = {
    "simple":        ("simple_split/tasks_train_simple.txt",
                      "simple_split/tasks_test_simple.txt"),
    "addprim_jump":  ("add_prim_split/tasks_train_addprim_jump.txt",
                      "add_prim_split/tasks_test_addprim_jump.txt"),
    "addprim_turn":  ("add_prim_split/tasks_train_addprim_turn_left.txt",
                      "add_prim_split/tasks_test_addprim_turn_left.txt"),
    "length":        ("length_split/tasks_train_length.txt",
                      "length_split/tasks_test_length.txt"),
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "scan_cache")


def _fetch(suffix: str) -> str:
    """Download `suffix` (cached under data/scan_cache/) and return local path."""
    local = os.path.join(CACHE_DIR, suffix.replace("/", "__"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(local):
        urllib.request.urlretrieve(_BASE + suffix, local)
    return local


def parse_line(line: str):
    """'IN: jump twice OUT: I_JUMP I_JUMP' → (['jump','twice'], ['I_JUMP','I_JUMP'])."""
    line = line.strip()
    assert line.startswith("IN:") and " OUT:" in line, f"malformed: {line!r}"
    in_part, out_part = line[len("IN:"):].split(" OUT:")
    return in_part.split(), out_part.split()


def _read_pairs(path: str):
    with open(path) as f:
        return [parse_line(ln) for ln in f if ln.strip()]


def build_vocab(pairs, index):
    """Vocab over token position `index` (0=input, 1=output). Reserves <pad>=0, <eos>=1."""
    toks = sorted({t for pair in pairs for t in pair[index]})
    stoi = {"<pad>": 0, "<eos>": 1}
    for t in toks:
        stoi[t] = len(stoi)
    return stoi


def load_scan(split: str = "addprim_jump"):
    """Returns dict: train/test pairs (token lists) + input/output vocabs + stats."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(SPLITS)}")
    train_suf, test_suf = SPLITS[split]
    train = _read_pairs(_fetch(train_suf))
    test  = _read_pairs(_fetch(test_suf))

    in_vocab  = build_vocab(train, 0)
    out_vocab = build_vocab(train, 1)
    max_in    = max(len(p[0]) for p in train + test)
    max_out   = max(len(p[1]) for p in train + test)

    return {
        "split": split,
        "train": train,
        "test": test,
        "in_vocab": in_vocab,
        "out_vocab": out_vocab,
        "max_in_len": max_in,
        "max_out_len": max_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="addprim_jump", choices=sorted(SPLITS))
    args = ap.parse_args()

    d = load_scan(args.split)
    print(f"SCAN split: {d['split']}")
    print(f"  train pairs : {len(d['train'])}")
    print(f"  test  pairs : {len(d['test'])}")
    print(f"  input vocab : {len(d['in_vocab'])}  (incl <pad>,<eos>)")
    print(f"  output vocab: {len(d['out_vocab'])}")
    print(f"  max in len  : {d['max_in_len']}")
    print(f"  max out len : {d['max_out_len']}")
    print(f"  example     : IN {d['train'][0][0]}  OUT {d['train'][0][1]}")
    # how many test commands contain a primitive composed in a novel way?
    if args.split == "addprim_jump":
        jump_test = sum(1 for p in d["test"] if "jump" in p[0])
        print(f"  test cmds containing 'jump': {jump_test}/{len(d['test'])}")


if __name__ == "__main__":
    main()
