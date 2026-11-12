"""Filter fragment SELFIES and write vocabulary-aligned label indices.

``--frags-jsonl`` supplies JSON objects with molecular ``selfies`` and a
``frag_selfies`` list. ``--frag-vocab`` supplies fragment SELFIES in index order.
The output JSONL retains input record order and filters fragments to this
vocabulary. Duplicate fragments are removed unless ``--no-dedup`` is selected.

``--out-labels-pkl`` contains a list of sorted unique vocabulary-index lists,
one per output JSONL record. Inputs must already use SELFIES; this module does
not convert molecular representations.
"""

import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple

def load_vocab(path: str) -> Tuple[List[str], Dict[str, int]]:
    """Read an ordered fragment vocabulary and its token-to-index mapping.

    Blank lines and lines beginning with ``#`` are ignored. Tokens are not
    converted between molecular representations. Repeated entries remain in
    the list, with the mapping pointing to their final occurrence.
    """
    toks: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            toks.append(t)
    tok2id = {t: i for i, t in enumerate(toks)}
    return toks, tok2id

def main():
    """Filter JSONL fragments to the vocabulary and write aligned label indices.

    Nonempty input-record order and unrelated JSON fields are preserved.
    Malformed JSON raises ValueError rather than silently losing a row.
    Labels are sorted unique indices even when ``--no-dedup`` preserves
    duplicate fragment text in the JSONL output. Input paths and both output
    paths must refer to distinct files.
    """
    ap = argparse.ArgumentParser("Fill fragments to vocab & build label indices")
    ap.add_argument("--frags-jsonl", required=True, help="input JSONL with frag_selfies")
    ap.add_argument("--frag-vocab", required=True, help="fragment vocab (SELFIES)")
    ap.add_argument("--out-jsonl", required=True, help="output filtered JSONL")
    ap.add_argument("--out-labels-pkl", required=True, help="output labels pickle")
    ap.add_argument("--dedup", action="store_true", help="deduplicate fragments per molecule (default on)")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false", help="disable deduplication")
    ap.set_defaults(dedup=True)
    args = ap.parse_args()

    inputs = [Path(args.frags_jsonl), Path(args.frag_vocab)]
    outputs = [Path(args.out_jsonl), Path(args.out_labels_pkl)]
    for index, output in enumerate(outputs):
        for other in inputs + outputs[:index]:
            if (output.resolve() == other.resolve()
                    or (output.exists() and other.exists() and output.samefile(other))):
                ap.error('Input files and output files must have distinct paths.')
        output.parent.mkdir(parents=True, exist_ok=True)

    vocab, tok2id = load_vocab(args.frag_vocab)

    n_rows = 0
    n_with_any_in_vocab = 0
    total_frags_in = 0
    total_frags_kept = 0
    oov_set: Set[str] = set()

    labels: List[List[int]] = []

    with open(args.frags_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue
            n_rows += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Malformed JSON at row {n_rows}; refusing to lose alignment.') from exc

            # Robust key access (we expect these keys from the extractor)
            mol_selfies = d.get("selfies", "")
            frag_list: List[str] = d.get("frag_selfies", []) or []

            total_frags_in += len(frag_list)

            kept_ids: List[int] = []
            kept_frags: List[str] = []

            if args.dedup:
                frag_iter = sorted(set(frag_list))
            else:
                frag_iter = frag_list

            for fs in frag_iter:
                if fs in tok2id:
                    kept_frags.append(fs)
                    kept_ids.append(tok2id[fs])
                else:
                    oov_set.add(fs)

            total_frags_kept += len(kept_frags)
            if len(kept_frags) > 0:
                n_with_any_in_vocab += 1

            # Update JSON and write out (preserve other fields if present)
            d["frag_selfies"] = kept_frags
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

            # Labels are sorted unique indices (stable for multi-hot)
            labels.append(sorted(set(kept_ids)))

    # Save labels
    with open(args.out_labels_pkl, "wb") as f:
        pickle.dump(labels, f)

    # Print summary
    pct_with = 100.0 * n_with_any_in_vocab / max(1, n_rows)
    avg_in = total_frags_in / max(1, n_rows)
    avg_kept = total_frags_kept / max(1, n_rows)
    print("=== fill_fragments summary ===")
    print(f"Rows processed:            {n_rows}")
    print(f"Rows with ≥1 in-vocab frag:{n_with_any_in_vocab} ({pct_with:.2f}%)")
    print(f"Avg fragments per row (in):   {avg_in:.3f}")
    print(f"Avg fragments per row (kept): {avg_kept:.3f}")
    print(f"Unique OOV fragments seen: {len(oov_set)}")
    print(f"Wrote JSONL:   {args.out_jsonl}")
    print(f"Wrote labels:  {args.out_labels_pkl}")

if __name__ == "__main__":
    main()
