"""Prepare canonical, deduplicated molecules with exact SELFIES round trips.

The CLI writes aligned SMILES/SELFIES files, an ordered token vocabulary,
frequency and exclusion CSVs, and a preprocessing report with length statistics.
"""
import os, json, csv, argparse, collections, math
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# RDKit import guard
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as Descriptors
except Exception as e:
    raise RuntimeError("RDKit is required. Install rdkit-pypi or conda rdkit.") from e

# SELFIES
try:
    import selfies as sf
except Exception as e:
    raise RuntimeError("selfies is required. pip install selfies") from e

import numpy as np


def mkdir_p(path: str):
    """Create an output directory and any missing parents if needed."""
    os.makedirs(path, exist_ok=True)


def largest_fragment_smiles(smiles: str) -> str:
    """Select the input salt component with most heavy atoms, keeping first ties.

    The returned component is not canonicalized here. A single-component input
    is returned unchanged; unparseable components are skipped during selection.
    """
    parts = smiles.split(".")
    if len(parts) == 1:
        return smiles
    max_size = -1
    best = parts[0]
    for p in parts:
        mol = Chem.MolFromSmiles(p)
        if mol is None:
            continue
        ha = mol.GetNumHeavyAtoms()
        if ha > max_size:
            max_size = ha
            best = p
    return best


def canonicalize_smiles(smiles: str, stereo: bool = True, salt_split: bool = True) -> Optional[str]:
    """Canonicalize a molecule, optionally retaining stereochemistry and one salt component.

    ``salt_split=True`` selects the largest component before canonicalization.
    Return None when parsing or canonical SMILES generation fails.
    """
    if salt_split:
        smiles = largest_fragment_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        can = Chem.MolToSmiles(mol, isomericSmiles=stereo, canonical=True)
        return can
    except Exception:
        return None


def selfies_roundtrip_ok(smiles: str, stereo: bool = True) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check molecular equivalence after a SELFIES encode/decode round trip.

    Parameters
    ----------
    smiles : str
        Molecular SMILES to encode.
    stereo : bool, default=True
        Whether canonical comparison preserves stereochemistry.

    Returns
    -------
    ok : bool
        Whether input and decoded canonical SMILES agree.
    selfies : str or None
        Encoded SELFIES when available.
    decoded_canonical_smiles : str or None
        Decoded molecule in the selected canonical representation, if valid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (False, None, None)
    try:
        # encode to selfies
        selfies = sf.encoder(smiles)
        if selfies is None:
            return (False, None, None)
        # decode back
        decoded_smiles = sf.decoder(selfies)
        if decoded_smiles is None:
            return (False, selfies, None)
        # canonicalize both sides and compare
        smi_can = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=stereo, canonical=True)
        dec_can_mol = Chem.MolFromSmiles(decoded_smiles)
        if dec_can_mol is None:
            return (False, selfies, None)
        dec_can = Chem.MolToSmiles(dec_can_mol, isomericSmiles=stereo, canonical=True)
        ok = (smi_can == dec_can)
        return (ok, selfies, dec_can)
    except Exception:
        return (False, None, None)


def tokenize_selfies(selfies_str: str) -> List[str]:
    """Split SELFIES into bracketed tokens, returning an empty list on failure."""
    try:
        return list(sf.split_selfies(selfies_str))
    except Exception:
        return []


def pct(v: float) -> float:
    """Convert a fraction to percent, rounded to four decimal places."""
    return float(round(v * 100.0, 4))


def build_vocab_from_tokenized(all_tokens: List[List[str]], min_freq_rare: int) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """Build a frequency-ordered vocabulary and report rare token counts.

    Parameters
    ----------
    all_tokens : list of list of str
        Tokenized molecules; repeated tokens contribute repeated counts.
    min_freq_rare : int
        Tokens with strictly fewer occurrences are reported as rare.

    Returns
    -------
    vocab : list of str
        PAD/BOS/EOS/UNK followed by tokens in ``Counter.most_common`` order.
    token_freq, rare : dict
        Counts for all observed tokens and for the rare subset. Rare tokens
        remain in the vocabulary.
    """
    counter = collections.Counter()
    for toks in all_tokens:
        counter.update(toks)
    # specials on top
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"]
    for tok, _ in counter.most_common():
        vocab.append(tok)
    token_freq = dict(counter)
    rare = {t: c for t, c in token_freq.items() if c < min_freq_rare}
    return vocab, token_freq, rare


def quantiles_len(seq_lengths: List[int]) -> Dict[str, int]:
    """Return nearest-observation p50/p90/p95/p99 and max; use zeros if empty."""
    if not seq_lengths:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    arr = np.array(seq_lengths, dtype=np.int32)
    return {
        "p50": int(np.quantile(arr, 0.50, interpolation="nearest")),
        "p90": int(np.quantile(arr, 0.90, interpolation="nearest")),
        "p95": int(np.quantile(arr, 0.95, interpolation="nearest")),
        "p99": int(np.quantile(arr, 0.99, interpolation="nearest")),
        "max": int(arr.max()),
    }


def write_list(path: str, lines: List[str]):
    """Write strings in input order, one per UTF-8 line, replacing the target file."""
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


def write_vocab(path: str, vocab: List[str]):
    """Write a vocabulary in index order, preserving its special-token prefix."""
    write_list(path, vocab)


def write_token_freq_csv(path: str, token_freq: Dict[str, int]):
    """Write token/count rows ordered by descending count, then token text."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token", "count"])
        for tok, cnt in sorted(token_freq.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([tok, cnt])


def write_excluded_csv(path: str, rows: List[Tuple[int, str, str]]):
    """Write exclusion records ``(one_based_stage_row, input_text, reason)``.

    Canonicalization records refer to nonempty input rows; round-trip records
    refer to the deduplicated canonical molecule list.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "input", "reason"])
        for r in rows:
            w.writerow(list(r))


def parse_args():
    """Parse preprocessing paths, stereochemistry policy, and reporting thresholds."""
    p = argparse.ArgumentParser("SELFIES sanity checker: canonicalize, round-trip, vocab, stats")
    p.add_argument("--input-txt", type=str, required=True, help="One SMILES per line")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    p.add_argument("--stereo", action="store_true", help="Keep isomeric info in canonicalization")
    p.add_argument("--no-stereo", dest="stereo", action="store_false", help="Drop stereochemistry")
    p.add_argument("--salt-split", action="store_true", help="Keep largest fragment from salts")
    p.add_argument("--roundtrip-target", type=float, default=0.995, help="Target exact round-trip rate")
    p.add_argument("--rare-token-min-freq", type=int, default=10, help="Tokens with freq < this are considered rare (reported)")
    p.set_defaults(stereo=True, salt_split=True)
    return p.parse_args()


def main():
    """Filter input molecules and write aligned data, vocabulary, and diagnostics."""
    args = parse_args()
    mkdir_p(args.outdir)

    # Read input
    with open(args.input_txt, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    n_raw = len(raw)

    # Canonicalize (+ salt split)
    canon = []
    excl = []  # (line_idx, raw, reason)
    seen = set()

    for i, s in enumerate(tqdm(raw, desc="Canonicalizing")):
        c = canonicalize_smiles(s, stereo=args.stereo, salt_split=args.salt_split)
        if c is None:
            excl.append((i + 1, s, "canonicalize_failed"))
            continue
        if c in seen:
            # de-dup
            continue
        seen.add(c)
        canon.append(c)

    n_canon_unique = len(canon)

    # SELFIES round-trip check
    selfies_ok = []
    selfies_tokens = []
    ok_flags = []
    decoded_canons = []
    roundtrip_excl = []

    for i, s in enumerate(tqdm(canon, desc="SELFIES round-trip")):
        ok, selfies_str, dec_can = selfies_roundtrip_ok(s, stereo=args.stereo)
        if not ok or selfies_str is None or dec_can is None:
            roundtrip_excl.append((i + 1, s, "roundtrip_canon_failed"))
            continue
        selfies_ok.append(selfies_str)
        decoded_canons.append(dec_can)
        toks = tokenize_selfies(selfies_str)
        selfies_tokens.append(toks)
        ok_flags.append(ok)

    n_roundtrip = len(selfies_ok)
    roundtrip_rate = (n_roundtrip / float(max(1, n_canon_unique)))

    # Build aligned outputs: SMILES must be dec_can (post-roundtrip canonical)
    clean_smiles = decoded_canons
    clean_selfies = selfies_ok

    # Build vocab & token freqs
    vocab, token_freq, rare = build_vocab_from_tokenized(selfies_tokens, args.rare_token_min_freq)

    # Length stats
    lengths = [len(toks) for toks in selfies_tokens]
    lens = quantiles_len(lengths)
    recommended_max_len = lens["p95"]

    # Write outputs
    out_smiles = os.path.join(args.outdir, "clean_smiles.txt")
    out_selfies = os.path.join(args.outdir, "clean_selfies.txt")
    out_vocab = os.path.join(args.outdir, "selfies_vocab.txt")
    out_tokfreq = os.path.join(args.outdir, "token_freq.csv")
    out_excl = os.path.join(args.outdir, "excluded_log.csv")
    out_report = os.path.join(args.outdir, "sanity_report.json")

    write_list(out_smiles, clean_smiles)
    write_list(out_selfies, clean_selfies)
    write_vocab(out_vocab, vocab)
    write_token_freq_csv(out_tokfreq, token_freq)
    write_excluded_csv(out_excl, excl + roundtrip_excl)

    report = {
        "input_file": os.path.abspath(args.input_txt),
        "n_input_rows": n_raw,
        "stereo": bool(args.stereo),
        "salt_split": bool(args.salt_split),
        "n_canonical_unique": n_canon_unique,
        "n_roundtrip_exact": n_roundtrip,
        "roundtrip_exact_canonical_rate": roundtrip_rate,
        "roundtrip_target": float(args.roundtrip_target),
        "passed_roundtrip_target": bool(roundtrip_rate >= args.roundtrip_target),
        "token_count": len(vocab),
        "num_rare_tokens": len(rare),
        "rare_token_min_freq": int(args.rare_token_min_freq),
        "length_stats": lens,
        "recommended_max_len": int(recommended_max_len),
        "notes": "Use recommended_max_len as --max-len for training. Vocab must keep specials on top.",
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Console summary
    print("\n=== SELFIES Sanity Report ===")
    print(f"Input rows:              {n_raw}")
    print(f"Canonical unique:        {n_canon_unique}")
    print(f"Round-trip exact (canon):{n_roundtrip}  ->  {pct(roundtrip_rate)}%")
    print(f"Vocab size:              {len(vocab)} (incl. specials)")
    print(f"Rare tokens (<{args.rare_token_min_freq}): {len(rare)}")
    print(f"Length p50/p90/p95/p99/max: {lens['p50']}/{lens['p90']}/{lens['p95']}/{lens['p99']}/{lens['max']}")
    print(f"Recommended --max-len:   {recommended_max_len}")
    print(f"\nOutputs in: {os.path.abspath(args.outdir)}")
    print("Files: clean_smiles.txt, clean_selfies.txt, selfies_vocab.txt, token_freq.csv, excluded_log.csv, sanity_report.json")


if __name__ == "__main__":
    main()
