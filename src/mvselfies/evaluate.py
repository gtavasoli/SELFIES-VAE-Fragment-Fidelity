#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate training and generated fragment distributions with BRICS.

Fragmentation uses spawn-based multiprocessing and a bounded result queue.
Each worker reuses a helper process and restarts it after a molecular timeout.
Duplicate input molecules contribute repeated fragment counts. Fragments can be
represented as SMILES or SELFIES. The labelled convention retains the BRICS
attachment labels used by the main 10k and 100k runs.

Output files in ``--out-dir`` are:

* ``frag_freq_train.csv`` and ``frag_freq_gen.csv``: fragment representation,
  occurrence count, and normalized probability.
* ``frag_freq_merged.csv``: fragment, count_train, p_train, count_gen, p_gen,
  and generated-minus-training probability delta_p.
* ``frag_eval_report.json``: JSD, rare recall, top-k mass, and OOV shares.
"""

import os
import re
import sys
import json
import math
import time
import argparse
import multiprocessing as mp
import queue
from typing import Optional, List, Tuple, Dict, Iterable, Set
from collections import Counter
from tqdm import tqdm

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

# Optional SELFIES (only if --space selfies)
try:
    import selfies as sf
except Exception:
    sf = None

# --------- BRICS helpers ---------

_CAP_RE = re.compile(r"\[(?:\d+\*|\*(?::\d+)?)\]")  # [*], [*:1], [14*], ...

def canonical_smiles(s: str) -> Optional[str]:
    """Return canonical isomeric SMILES, or None for parsing/sanitization failure."""
    try:
        m = Chem.MolFromSmiles(s, sanitize=True)
        if not m:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None

def cap_brics_fragment_smiles(frag_smi: str, fragment_convention: str = "capped") -> Optional[str]:
    """Apply the selected attachment-atom convention and canonicalize a fragment.

    ``capped`` replaces labelled and unlabelled dummy atoms with hydrogen.
    ``labelled`` preserves isotope-labelled attachment atoms such as ``[16*]``
    while replacing unlabelled dummies. Invalid output returns None.
    """
    try:
        pattern = _CAP_RE if fragment_convention == "capped" else re.compile(r"\[\*(?::\d+)?\]")
        capped = pattern.sub("[H]", frag_smi)
        return canonical_smiles(capped)
    except Exception:
        return None

def brics_fragments_smiles(
    smi: str,
    min_heavy_atoms: int,
    max_frags_per_mol: int,
    fragment_convention: str = "capped",
) -> List[str]:
    """Return unique canonical BRICS fragments in a deterministic largest-first order.

    Parameters
    ----------
    smi : str
        Molecular SMILES; invalid molecules produce an empty list.
    min_heavy_atoms : int
        Minimum fragment heavy-atom count after applying the selected convention.
    max_frags_per_mol : int
        Positive limit on retained fragments per molecule.
    fragment_convention : {'capped', 'labelled'}, default='capped'
        Attachment-atom representation used by the archived count distributions.

    Returns
    -------
    list of str
        Unique fragment SMILES sorted by descending heavy-atom count, then text.
        Fragmentation failures or no eligible fragments produce an empty list.
    """
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            return []
        raw_frags: Set[str] = BRICS.BRICSDecompose(mol, minFragmentSize=1)
        if not raw_frags:
            return []
        sized: List[Tuple[int,str]] = []
        for fs in raw_frags:
            can = cap_brics_fragment_smiles(fs, fragment_convention)
            if not can:
                continue
            m = Chem.MolFromSmiles(can)
            if not m:
                continue
            hv = m.GetNumHeavyAtoms()
            if hv >= min_heavy_atoms:
                sized.append((hv, can))
        if not sized:
            return []
        sized.sort(key=lambda x: (-x[0], x[1]))
        seen, keep = set(), []
        for _, s in sized:
            if s not in seen:
                seen.add(s)
                keep.append(s)
            if len(keep) >= max_frags_per_mol:
                break
        return keep
    except Exception:
        return []

def smiles_to_selfies(smi: str) -> Optional[str]:
    """Encode fragment SMILES, or return None if SELFIES is unavailable or fails."""
    if sf is None:
        return None
    try:
        return sf.encoder(smi)
    except Exception:
        return None

# --------- IO ---------

def iter_raw_smiles(path: str) -> Iterable[str]:
    """Yield nonempty stripped lines, preserving duplicate molecule occurrences."""
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s:
                yield s

def load_vocab(path: Optional[str]) -> Optional[Set[str]]:
    """Read nonempty vocabulary lines as a set, or return None for a missing path."""
    if not path or not os.path.exists(path):
        return None
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            t = ln.strip()
            if t:
                out.add(t)
    return out

# --------- Windows-safe helper/runner ---------

def _runner_loop(conn, min_heavy_atoms: int, max_frags_per_mol: int, fragment_convention: str):
    """Serve fragmentation requests in a persistent helper process.

    Pipe messages are ``('SMI', smiles)`` and ``('STOP', None)``. Each molecular
    request receives ``('OK', fragments)``; fragmentation failures yield an
    empty list. The pipe is closed on stop or connection failure.
    """
    try:
        while True:
            msg = conn.recv()
            if not isinstance(msg, tuple) or len(msg) != 2:
                continue
            tag, payload = msg
            if tag == 'STOP':
                break
            if tag != 'SMI':
                continue
            smi = payload
            try:
                fr = brics_fragments_smiles(smi, min_heavy_atoms, max_frags_per_mol, fragment_convention)
            except Exception:
                fr = []
            try:
                conn.send(('OK', fr))
            except Exception:
                pass
    except (EOFError, OSError):
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _start_runner(ctx: mp.context.BaseContext, min_heavy_atoms: int, max_frags_per_mol: int, fragment_convention: str):
    """Start a non-daemon helper and return its process plus parent pipe endpoint."""
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    p = ctx.Process(target=_runner_loop, args=(child_conn, min_heavy_atoms, max_frags_per_mol, fragment_convention))
    p.daemon = False
    p.start()
    return p, parent_conn

def _frag_worker_chunk(
    chunk: List[str],
    out_q: "mp.Queue",
    min_heavy_atoms: int,
    max_frags_per_mol: int,
    frag_timeout_sec: float,
    fragment_convention: str,
):
    """Process one chunk using a restartable helper and stream queue messages.

    ``frag_timeout_sec`` is a per-molecule wall-time limit in seconds. Invalid
    or timed-out molecules emit ``('ITEM', [])``; successful items emit fragment
    lists. Infrastructure failures emit ``('ERROR', description)``. The worker
    finishes with ``('DONE', None)`` and closes its helper. Duplicate molecular
    lines are processed separately.
    """
    start_method = "spawn" if sys.platform == "win32" else "fork"
    ctx = mp.get_context(start_method)

    runner, conn = _start_runner(ctx, min_heavy_atoms, max_frags_per_mol, fragment_convention)

    def restart():
        nonlocal runner, conn
        try:
            if runner.is_alive():
                try:
                    conn.send(('STOP', None))
                except Exception:
                    pass
                runner.join(timeout=0.2)
        except Exception:
            pass
        if runner.is_alive():
            try: runner.terminate()
            except Exception: pass
            runner.join(timeout=0.5)
        try:
            conn.close()
        except Exception:
            pass
        runner, conn = _start_runner(ctx, min_heavy_atoms, max_frags_per_mol, fragment_convention)

    try:
        for smi in chunk:
            # canonicalize here; do NOT dedupe (duplicates should count)
            can = canonical_smiles(smi)
            if not can:
                out_q.put(('ITEM', []))
                continue
            try:
                conn.send(('SMI', can))
            except Exception:
                restart()
                out_q.put(('ITEM', []))
                continue

            t0 = time.time()
            sent = False
            while True:
                if conn.poll(0.01):
                    try:
                        tag, fr = conn.recv()
                        if tag != 'OK':
                            fr = []
                    except Exception:
                        fr = []
                    out_q.put(('ITEM', fr))
                    sent = True
                    break
                if (time.time() - t0) >= frag_timeout_sec:
                    restart()
                    out_q.put(('ITEM', []))
                    sent = True
                    break
            if not sent:
                out_q.put(('ITEM', []))
    except Exception as exc:
        out_q.put(('ERROR', repr(exc)))
    finally:
        try:
            conn.send(('STOP', None))
        except Exception:
            pass
        try:
            runner.join(timeout=0.2)
        except Exception:
            pass
        if runner.is_alive():
            try: runner.terminate()
            except Exception: pass
            runner.join(timeout=0.5)
        try:
            conn.close()
        except Exception:
            pass
        out_q.put(('DONE', None))

def _chunks_from_file(path: str, chunk_size: int) -> Iterable[List[str]]:
    """Yield lists of at most ``chunk_size`` nonempty lines without loading the file."""
    buf: List[str] = []
    for s in iter_raw_smiles(path):
        buf.append(s)
        if len(buf) >= chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf

def _bounded_controller_build_counter(
    smiles_path: str,
    *,
    min_heavy_atoms: int,
    max_frags_per_mol: int,
    n_workers: int,
    chunk_size: int,
    queue_mult: int,
    frag_timeout_sec: float,
    desc: str,
    fragment_convention: str = "capped",
) -> Tuple[Counter, Dict[str, float]]:
    """Accumulate fragment occurrences with bounded worker/queue fan-out.

    Parameters
    ----------
    smiles_path : str
        Input molecule file; duplicate lines retain their statistical weight.
    min_heavy_atoms, max_frags_per_mol : int
        Fragment size filter and per-molecule retained-fragment limit.
    n_workers, chunk_size, queue_mult : int
        Concurrent chunk-worker limit, molecules per chunk, and queue multiplier.
        Queue capacity is ``max(2*chunk_size, queue_mult*chunk_size)``.
    frag_timeout_sec : float
        Per-molecule timeout in seconds; timed-out molecules have no fragments.
    desc : str
        Progress-bar label.
    fragment_convention : {'capped', 'labelled'}
        Attachment-atom policy used before counting.

    Returns
    -------
    counts : collections.Counter
        Fragment occurrences accumulated from received molecular results.
    stats : dict
        Received molecule count, count/percentage with fragments, and population
        mean/standard deviation of fragment counts. Percentages use 0 to 100.

    Raises
    ------
    RuntimeError
        If workers fail, exit without completing the queue protocol, or return
        fewer molecular results than the input contains.
    """
    start_method = "spawn" if sys.platform == "win32" else "fork"
    ctx = mp.get_context(start_method)
    q_maxsize = max(2*chunk_size, queue_mult*chunk_size)
    out_q: mp.Queue = ctx.Queue(maxsize=q_maxsize)

    # pre-scan for progress length (optional, cheap per-line count)
    n_lines = sum(1 for _ in iter_raw_smiles(smiles_path))

    chunks = _chunks_from_file(smiles_path, chunk_size)
    procs: List[mp.Process] = []
    active = 0

    def start_next() -> bool:
        nonlocal active
        try:
            ch = next(chunks)
        except StopIteration:
            return False
        p = ctx.Process(
            target=_frag_worker_chunk,
            args=(ch, out_q, min_heavy_atoms, max_frags_per_mol, frag_timeout_sec, fragment_convention),
        )
        p.daemon = False
        p.start()
        procs.append(p)
        active += 1
        return True

    for _ in range(n_workers):
        if not start_next():
            break

    ctr = Counter()
    n_mols = 0
    n_with = 0
    fsum = 0.0
    fsum2 = 0.0

    with tqdm(total=n_lines, desc=desc, unit="mol", dynamic_ncols=True) as pbar:
        try:
            while active > 0:
                try:
                    tag, payload = out_q.get(timeout=1.0)
                except queue.Empty:
                    if not any(p.is_alive() for p in procs):
                        raise RuntimeError('Fragment workers exited without completing their input.')
                    continue
                if tag == 'ITEM':
                    fr = payload or []
                    n_mols += 1
                    k = len(fr)
                    fsum += k
                    fsum2 += (k*k)
                    if k:
                        n_with += 1
                        ctr.update(fr)
                    pbar.update(1)
                elif tag == 'ERROR':
                    raise RuntimeError(f'Fragment worker failed: {payload}')
                elif tag == 'DONE':
                    active -= 1
                    # reap finished
                    for p in list(procs):
                        if not p.is_alive():
                            p.join(timeout=0.1)
                            procs.remove(p)
                    start_next()
        finally:
            # ensure cleanup
            for p in procs:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=1.0)

    if n_mols != n_lines:
        raise RuntimeError(
            f'Fragmentation processed {n_mols} of {n_lines} molecules; refusing a partial report.'
        )
    mean = float(fsum / max(1, n_mols))
    var  = float((fsum2 / max(1, n_mols)) - mean*mean)
    std  = float(math.sqrt(max(0.0, var)))

    stats = {
        "n_molecules": int(n_mols),
        "n_with_any_fragment": int(n_with),
        "pct_with_any_fragment": 100.0 * n_with / max(1, n_mols),
        "frags_per_molecule_mean": mean,
        "frags_per_molecule_std": std,
    }
    return ctr, stats

# --------- metrics / CSV / JSON ---------

def jsd_full(p_counts: Counter, q_counts: Counter) -> float:
    """Compute Jensen-Shannon distance over the union of fragment keys.

    Parameters
    ----------
    p_counts, q_counts : collections.Counter
        Nonnegative fragment occurrence counts, normalized independently here.

    Returns
    -------
    float
        Square root of Jensen-Shannon divergence using natural logarithms.
        Probability terms are clipped at 1e-12 for numerical stability.

    Raises
    ------
    ValueError
        If either distribution has no positive total count.
    """
    # build aligned p, q arrays
    if sum(p_counts.values()) <= 0 or sum(q_counts.values()) <= 0:
        raise ValueError('Both fragment distributions must contain positive counts.')
    keys = list(set(p_counts.keys()) | set(q_counts.keys()))
    p = np.array([p_counts.get(k,0) for k in keys], dtype=np.float64)
    q = np.array([q_counts.get(k,0) for k in keys], dtype=np.float64)
    eps = 1e-12
    p = p / max(eps, p.sum())
    q = q / max(eps, q.sum())
    m = 0.5*(p+q)
    def _kl(a,b):
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return float(np.sum(a*(np.log(a)-np.log(b))))
    jsd2 = 0.5*_kl(p,m) + 0.5*_kl(q,m)
    return float(math.sqrt(max(0.0, jsd2)))

def topk_coverage(counter: Counter, ks=(10,50,100,500,1000)) -> Dict[int,float]:
    """Return probability mass in a distribution's own top-k fragments.

    ``ks`` must be nondecreasing because cumulative counts are reused. Empty
    input gives zero mass; k beyond the support includes all observed counts.
    """
    items = counter.most_common()
    total = sum(counter.values()) or 1
    cov = {}
    cum = 0; i = 0
    for k in ks:
        while i < min(k, len(items)):
            cum += items[i][1]; i += 1
        cov[int(k)] = float(cum/total)
    return cov

def write_side_csv(path: str, colname: str, ctr: Counter):
    """Write fragment/count/probability rows in descending count order.

    ``colname`` names the fragment representation column; the remaining column
    names are ``count`` and ``p``. The target file is replaced.
    """
    total = sum(ctr.values()) or 1
    items = ctr.most_common()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{colname},count,p\n")
        for frag, cnt in items:
            p = float(cnt / total)
            f.write(f"{frag},{cnt},{p}\n")

def write_merged_csv(path: str, colname: str, ctrT: Counter, ctrG: Counter):
    """Write union-support counts, probabilities, and generated-minus-training mass."""
    totalT = sum(ctrT.values()) or 1
    totalG = sum(ctrG.values()) or 1
    keys = sorted(set(ctrT.keys()) | set(ctrG.keys()))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{colname},count_train,p_train,count_gen,p_gen,delta_p\n")
        for k in keys:
            ct = ctrT.get(k, 0); pt = float(ct / totalT)
            cg = ctrG.get(k, 0); pg = float(cg / totalG)
            f.write(f"{k},{ct},{pt},{cg},{pg},{pg-pt}\n")

def quantile_q10_train(counter: Counter) -> int:
    """Return the positive-count threshold covering at least 10% of fragment types.

    The target rank is ``ceil(0.1*n_positive_types)``. All count ties at the
    threshold are included, so the rare set can exceed 10%. Empty input gives 0.
    """
    hist = {}
    n = 0
    for _, c in counter.items():
        if c > 0:
            hist[c] = hist.get(c, 0) + 1
            n += 1
    if n == 0:
        return 0
    target = int(math.ceil(0.10 * n))
    acc = 0
    for c in sorted(hist.keys()):
        acc += hist[c]
        if acc >= target:
            return c
    return max(hist.keys())

def compute_rare_recall(train_ctr: Counter, gen_ctr: Counter) -> float:
    """Return the fraction of rare training fragment types observed in generation.

    Rarity uses ``quantile_q10_train`` with all threshold ties included. Types
    contribute equally regardless of counts. No rare training types gives 0.0.
    """
    q10 = quantile_q10_train(train_ctr)
    if q10 == 0:
        return 0.0
    rare = [k for k,v in train_ctr.items() if 0 < v <= q10]
    if not rare:
        return 0.0
    hit = sum(1 for r in rare if gen_ctr.get(r,0) > 0)
    return float(hit / max(1, len(rare)))

def restricted_vocab_metrics(vocab: Optional[Set[str]], ctrT: Counter, ctrG: Counter) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Measure restricted-support JSD and out-of-vocabulary occurrence shares.

    Parameters
    ----------
    vocab : set of str or None
        Allowed fragments in the same representation as the count keys.
    ctrT, ctrG : collections.Counter
        Training and generated occurrence counts.

    Returns
    -------
    jsd : float or None
        Natural-log Jensen-Shannon distance after renormalizing in-vocabulary
        counts; None if either side has no retained mass.
    oov_train, oov_gen : float or None
        Fractions of occurrence mass outside the supplied vocabulary. All three
        values are None if the vocabulary is absent or empty.
    """
    if not vocab:
        return None, None, None
    PT = sum(v for k,v in ctrT.items() if k in vocab)
    PG = sum(v for k,v in ctrG.items() if k in vocab)
    if PT == 0 or PG == 0:
        return None, 1.0 - float(PT / max(1, sum(ctrT.values()))), 1.0 - float(PG / max(1, sum(ctrG.values())))
    # build aligned arrays over vocab only
    keys = [k for k in ctrT.keys() | ctrG.keys() if k in vocab]
    p = np.array([ctrT.get(k,0) for k in keys], dtype=np.float64); p = p / max(1.0, p.sum())
    q = np.array([ctrG.get(k,0) for k in keys], dtype=np.float64); q = q / max(1.0, q.sum())
    eps = 1e-12
    m = 0.5*(p+q)
    def _kl(a,b):
        a = np.clip(a, eps, 1.0); b = np.clip(b, eps, 1.0)
        return float(np.sum(a*(np.log(a)-np.log(b))))
    jsd2 = 0.5*_kl(p,m) + 0.5*_kl(q,m)
    jsd_v = float(math.sqrt(max(0.0, jsd2)))
    oovT = 1.0 - float(PT / max(1, sum(ctrT.values())))
    oovG = 1.0 - float(PG / max(1, sum(ctrG.values())))
    return jsd_v, oovT, oovG

# --------- CLI / main ---------

def parse_args() -> argparse.Namespace:
    """Parse input paths, fragment conventions, worker limits, and metric options."""
    ap = argparse.ArgumentParser("Fragment fidelity (Windows-friendly, bounded memory)")
    ap.add_argument("--gen", required=True, help="Generated SMILES (one per line)")
    ap.add_argument("--train", required=True, help="Training SMILES (one per line)")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--fragment-convention", choices=["capped", "labelled"], default="capped",
                    help="Main 10k/100k runs used labelled fragments; 1M used capped fragments.")
    ap.add_argument("--space", choices=["selfies","smiles"], default="smiles",
                    help="Fragment representation; SMILES is faster.")
    ap.add_argument("--min-heavy-atoms", type=int, default=5)
    ap.add_argument("--max-frags-per-mol", type=int, default=6)

    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3),
                    help="Worker processes (each handles one chunk then exits).")
    ap.add_argument("--chunk-size", type=int, default=400, help="Molecules per worker chunk.")
    ap.add_argument("--queue-mult", type=int, default=4, help="Queue capacity multiplier.")
    ap.add_argument("--frag-timeout-sec", type=float, default=60.0, help="Timeout per molecule in seconds.")

    ap.add_argument("--frag-vocab", type=str, default=None, help="Restricted vocabulary (same space).")
    ap.add_argument("--progress", action="store_true", help="Show detailed progress bars.")
    return ap.parse_args()

def ensure_dir(path: str):
    """Create the parent directory of an output file path if one is specified."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _to_json_atomic(obj: dict, path: str):
    """Write JSON via a sibling temporary file and replace the destination atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def main():
    """Count train/generated fragments and write frequency CSVs plus a metric report."""
    args = parse_args()
    if args.space == "selfies" and sf is None:
        raise SystemExit("SELFIES space requested but `selfies` is not installed.")

    os.makedirs(args.out_dir, exist_ok=True)
    colname = "fragment_selfies" if args.space == "selfies" else "fragment_smiles"

    # ----- TRAIN -----
    train_ctr, train_stats = _bounded_controller_build_counter(
        args.train,
        min_heavy_atoms=args.min_heavy_atoms,
        max_frags_per_mol=args.max_frags_per_mol,
        n_workers=max(1, int(args.workers)),
        chunk_size=max(80, int(args.chunk_size)),
        queue_mult=max(2, int(args.queue_mult)),
        frag_timeout_sec=float(args.frag_timeout_sec),
        fragment_convention=args.fragment_convention,
        desc=f"BRICS→SMILES [train]" if args.progress else "train",
    )

    # Convert to requested space (only at the end, to save time if SMILES is chosen)
    if args.space == "selfies":
        train_ctr_conv = Counter()
        for k, v in train_ctr.items():
            sel = smiles_to_selfies(k)
            if sel:
                train_ctr_conv[sel] += v
        train_ctr = train_ctr_conv

    # ----- GEN -----
    gen_ctr, gen_stats = _bounded_controller_build_counter(
        args.gen,
        min_heavy_atoms=args.min_heavy_atoms,
        max_frags_per_mol=args.max_frags_per_mol,
        n_workers=max(1, int(args.workers)),
        chunk_size=max(80, int(args.chunk_size)),
        queue_mult=max(2, int(args.queue_mult)),
        frag_timeout_sec=float(args.frag_timeout_sec),
        fragment_convention=args.fragment_convention,
        desc=f"BRICS→SMILES [gen]" if args.progress else "gen",
    )
    if args.space == "selfies":
        gen_ctr_conv = Counter()
        for k, v in gen_ctr.items():
            sel = smiles_to_selfies(k)
            if sel:
                gen_ctr_conv[sel] += v
        gen_ctr = gen_ctr_conv

    # ----- Metrics -----
    jsd_val = jsd_full(train_ctr, gen_ctr)
    rare_recall = compute_rare_recall(train_ctr, gen_ctr)
    cov_train = topk_coverage(train_ctr)
    cov_gen   = topk_coverage(gen_ctr)

    vocab = load_vocab(args.frag_vocab)
    jsd_v, oovT, oovG = restricted_vocab_metrics(vocab, train_ctr, gen_ctr)

    # ----- CSVs -----
    out_train_csv  = os.path.join(args.out_dir, "frag_freq_train.csv")
    out_gen_csv    = os.path.join(args.out_dir, "frag_freq_gen.csv")
    out_merged_csv = os.path.join(args.out_dir, "frag_freq_merged.csv")
    write_side_csv(out_train_csv, colname, train_ctr)
    write_side_csv(out_gen_csv,   colname, gen_ctr)
    write_merged_csv(out_merged_csv, colname, train_ctr, gen_ctr)

    # ----- Report -----
    report = {
        "space": args.space,
        "fragment_convention": args.fragment_convention,
        "train_stats": train_stats,
        "gen_stats": gen_stats,
        "jsd_full_vocab": jsd_val,
        "rare_fragment_recall": rare_recall,
        "coverage_train_topk": cov_train,
        "coverage_gen_topk": cov_gen,
        "restricted_vocab_path": args.frag_vocab,
        "jsd_restricted_vocab": jsd_v,
        "oov_share_train_restricted": oovT,
        "oov_share_gen_restricted": oovG,
        "params": {
            "min_heavy_atoms": args.min_heavy_atoms,
            "max_frags_per_mol": args.max_frags_per_mol,
            "workers": int(args.workers),
            "chunk_size": int(args.chunk_size),
            "queue_mult": int(args.queue_mult),
            "frag_timeout_sec": float(args.frag_timeout_sec),
        }
    }
    _to_json_atomic(report, os.path.join(args.out_dir, "frag_eval_report.json"))

    # ----- Print summary -----
    print("\n=== Fragment fidelity report (Windows-friendly) ===")
    print(f"[space={args.space}]")
    print(f"JSD (full): {jsd_val:.4f}")
    if jsd_v is not None:
        print(f"JSD (restricted vocab): {jsd_v:.4f}")
        print(f"OOV share (train/gen) within vocab: {oovT:.3f} / {oovG:.3f}")
    print(f"Rare-fragment recall: {rare_recall*100:.2f}%")
    print("Top-k coverage (train):", cov_train)
    print("Top-k coverage (gen):  ", cov_gen)
    print(f"CSV outputs in: {args.out_dir}")

if __name__ == "__main__":
    # Windows requires this guard for multiprocessing
    mp.freeze_support()
    main()
