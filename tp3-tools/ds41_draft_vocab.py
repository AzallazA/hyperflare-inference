#!/usr/bin/env python3
"""Frequency-ranked draft vocabulary for DS41's DSpark draft head (FR-Spec style; TP3Config.draft_vocab).

  ds41_draft_vocab.py TOKENIZER_JSON OUT.json --k 32768 TEXT_OR_JSONL...

Counts the tokens of the given texts (``.jsonl``: every string under ``choices[*].turns``, LiveBench model answers;
anything else: the whole file as text), keeps every special / added token, then the most frequent tokens, at most K in all; the recipe
fills each rank's vocabulary shard up to K/world columns with its lowest unused ids (BPE merge order, a frequency proxy
for text the corpus lacks). The draft head
proposes only these tokens; the target verifies every token with its full head, so outputs are unchanged and only
draft acceptance on rare tokens can drop. Writes {"k", "tokens" (sorted ids), "source", "coverage"}.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

VOCAB = 129280


def texts(path):
    if path.suffix == ".jsonl":
        for line in path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            for choice in row.get("choices", []):
                yield from (t for t in choice.get("turns", []) if isinstance(t, str))
    else:
        yield path.read_text(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--k", type=int, default=32768)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    special = {t["id"] for t in json.load(open(a.tokenizer)).get("added_tokens", []) if t["id"] < VOCAB}
    counts, total = collections.Counter(), 0
    for name in a.inputs:
        for text in texts(Path(name)):
            ids = tok.encode(text, add_special_tokens=False).ids
            counts.update(ids)
            total += len(ids)
    keep = set(special)
    for token, _ in counts.most_common():
        if len(keep) >= a.k:
            break
        keep.add(token)
    # The recipe fills every rank's share up to the same width with its lowest unused ids (draft_vocab_columns), so
    # only the corpus-ranked and special tokens are listed here.
    covered = sum(c for t, c in counts.items() if t in keep)
    out = {"k": a.k, "tokens": sorted(keep), "coverage": round(covered / max(total, 1), 6), "corpus_tokens": total,
           "distinct_corpus_tokens": len(counts), "special_tokens": len(special),
           "source": "ds41_draft_vocab.py: " + ", ".join(Path(n).name for n in a.inputs)}
    Path(a.out).write_text(json.dumps(out))
    print(f"DRAFT VOCAB k {a.k}: corpus {total:,} tokens ({len(counts):,} distinct), coverage {100 * out['coverage']:.3f}%, "
          f"special {len(special)}", file=sys.stderr)


if __name__ == "__main__":
    main()
