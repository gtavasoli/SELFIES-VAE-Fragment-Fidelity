"""Extract BRICS fragments in canonical molecule order.

Input molecules are canonicalized and deduplicated in first-occurrence order.
Outputs include JSONL fragment records, a fragment SELFIES vocabulary, and
optional occurrence counts.

Each chunk worker reuses a helper process for BRICS decomposition. Molecular
timeouts trigger a helper restart. A bounded result queue limits memory use,
and the controller restores input order before yielding records. Multiprocessing
supports Windows and POSIX without signal-based timeouts.
"""

import os
import re
import sys
import json
import time
import argparse
import multiprocessing as mp
import queue
from datetime import datetime as dt
from typing import Optional, List, Set, Tuple, Iterable
from collections import Counter

import selfies as sf
from rdkit import Chem
from rdkit.Chem import BRICS
from tqdm import tqdm


# ------------------------ utils ------------------------

def canonical_smiles(s: str) -> Optional[str]:
    """Return RDKit-canonical SMILES (isomeric) or None on failure."""
    try:
        m = Chem.MolFromSmiles(s, sanitize=True)
        if not m:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None


# Matches BRICS dummy atoms in fragments: [*], [*:1], [*:12], [14*]
_CAP_RE = re.compile(r"\[(?:\d+\*|\*(?::\d+)?)\]")


def cap_brics_fragment_smiles(frag_smi: str) -> Optional[str]:
    """Replace BRICS dummy atoms by [H], then canonicalize."""
    try:
        capped = _CAP_RE.sub("[H]", frag_smi)
        can = canonical_smiles(capped)
        return can
    except Exception:
        return None


def brics_fragments_from_smiles(
    smi: str,
    min_heavy_atoms: int = 5,
    max_frags_per_mol: int = 6,
) -> List[str]:
    """Return unique hydrogen-capped BRICS fragments, ordered largest first.

    Parameters
    ----------
    smi : str
        Molecular SMILES to decompose.
    min_heavy_atoms : int, default=5
        Minimum fragment heavy-atom count after replacing attachment dummies.
    max_frags_per_mol : int, default=6
        Positive limit on the number of retained fragments.

    Returns
    -------
    list of str
        Canonical fragment SMILES sorted by descending heavy-atom count, then
        text. Invalid molecules, failures, and no eligible fragments give [].
    """
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            return []
        raw_frags: Set[str] = BRICS.BRICSDecompose(mol, minFragmentSize=1)
        if not raw_frags:
            return []

        capped: List[Tuple[int, str]] = []
        for fs in raw_frags:
            can = cap_brics_fragment_smiles(fs)
            if not can:
                continue
            m = Chem.MolFromSmiles(can)
            if not m:
                continue
            heavy = m.GetNumHeavyAtoms()
            if heavy >= min_heavy_atoms:
                capped.append((heavy, can))

        if not capped:
            return []

        capped.sort(key=lambda x: (-x[0], x[1]))

        seen: Set[str] = set()
        keep: List[str] = []
        for _, s in capped:
            if s not in seen:
                seen.add(s)
                keep.append(s)
            if len(keep) >= max_frags_per_mol:
                break
        return keep
    except Exception:
        return []


def smiles_to_selfies(s: str) -> Optional[str]:
    """Convert SMILES to SELFIES; return None on failure."""
    try:
        return sf.encoder(s)
    except Exception as e:
        print(f'Exception Convert SMILES to SELFIES: [{e}]')
        return None


# ------------------------ IO ------------------------

def read_smiles(path: str) -> List[str]:
    """Read nonempty stripped SMILES lines, preserving order and duplicates."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def read_selfies_decode_to_smiles(path: str) -> List[str]:
    """Decode nonempty SELFIES lines to SMILES, skipping failed/empty decodings."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sel = line.strip()
            if not sel:
                continue
            try:
                smi = sf.decoder(sel)
            except Exception:
                smi = None
            if smi:
                out.append(smi)
    return out


# ------------------------ parallel canonicalization ------------------------

def _worker_canon(smi: str) -> Optional[str]:
    """Canonicalize one pool task, returning None instead of propagating failures."""
    try:
        return canonical_smiles(smi)
    except Exception:
        return None


def canonicalize_parallel(
    raw_smiles: List[str],
    *,
    chunksize: int = 2000,
    maxtasksperchild: int = 2000,
    desc: str = "Canonicalizing",
    workers: Optional[int] = None,
) -> List[str]:
    """Canonicalize in parallel and retain unique valid molecules in input order.

    Parameters
    ----------
    raw_smiles : list of str
        Molecules to canonicalize; invalid entries are dropped.
    chunksize, maxtasksperchild : int
        Pool task-batch size and task limit before a pool worker is replaced.
    desc : str
        Progress-bar label.
    workers : int, optional
        Requested worker count, capped by the available CPU budget.

    Returns
    -------
    list of str
        Canonical isomeric SMILES in first-occurrence order. Ordered pool
        iteration and parent-side deduplication keep the output deterministic
        for a fixed RDKit version and input sequence.
    """
    cpu_total = os.cpu_count() or 4
    n_workers = max(1, min(workers or cpu_total-2, cpu_total-2))

    canon_list: List[str] = []
    seen_smiles: Set[str] = set()

    # On Windows, use 'spawn'; on POSIX, 'fork' is fine but 'spawn' is okay too.
    start_method = "spawn" if sys.platform == "win32" else "fork"
    ctx = mp.get_context(start_method)

    with ctx.Pool(processes=n_workers, maxtasksperchild=maxtasksperchild) as pool:
        it = pool.imap(_worker_canon, raw_smiles, chunksize=chunksize)
        for can in tqdm(it, desc=desc, unit="mol"):
            if not can:
                continue
            if can not in seen_smiles:
                seen_smiles.add(can)
                canon_list.append(can)

    return canon_list


# ------------------------ helper child (BRICS runner) ------------------------

def _brics_runner_loop(conn, params):
    """Serve BRICS requests in a persistent child using a two-way pipe.

    The parent sends ``('SMI', smiles)`` or ``('STOP', None)``; the child replies
    ``('OK', fragments)``. ``params`` supplies ``min_heavy_atoms`` and
    ``max_frags_per_mol``. The worker owns timeout/restart decisions.
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
                frags_smiles = brics_fragments_from_smiles(
                    smi,
                    min_heavy_atoms=params['min_heavy_atoms'],
                    max_frags_per_mol=params['max_frags_per_mol'],
                )
            except Exception:
                frags_smiles = []
            conn.send(('OK', frags_smiles))
    except (EOFError, OSError):
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _start_brics_runner(ctx, params):
    """Start a BRICS child and return its process and parent pipe connection."""
    parent_conn, child_conn = mp.Pipe(duplex=True)
    p = ctx.Process(target=_brics_runner_loop, args=(child_conn, params))
    p.daemon = False
    p.start()
    # Child owns child_conn; parent keeps parent_conn
    return p, parent_conn


# ------------------------ fragmentation (bounded producer–consumer) ------------------------

# Queue messages: ('ITEM', smi, frags_smiles) or ('DONE', None, None)

def _frags_worker_stream(
    chunk: List[str],
    out_q: "mp.Queue",
    params: dict,
    frag_timeout_sec: int,
):
    """Stream one chunk through a helper with a per-molecule timeout in seconds.

    Queue items are ``('ITEM', smiles, fragments)`` followed by
    ``('DONE', None, None)``. A timeout or failed request produces an empty
    fragment list. Unexpected worker errors use the ``__EXC__:`` sentinel so
    the controller can fail rather than silently lose molecular alignment.
    """
    # Use same start method as the parent created us with
    start_method = "spawn" if sys.platform == "win32" else "fork"
    ctx = mp.get_context(start_method)

    # Start helper child (runner)
    helper_proc, conn = _start_brics_runner(ctx, params)

    def restart_helper():
        nonlocal helper_proc, conn
        try:
            if helper_proc.is_alive():
                # Try to stop cleanly
                try:
                    conn.send(('STOP', None))
                except Exception:
                    pass
                helper_proc.join(timeout=0.2)
        except Exception:
            pass
        # Hard kill if still alive
        if helper_proc.is_alive():
            try:
                helper_proc.terminate()
            except Exception:
                pass
            helper_proc.join(timeout=0.5)
        try:
            conn.close()
        except Exception:
            pass
        helper_proc, conn = _start_brics_runner(ctx, params)

    try:
        for smi in chunk:
            # Ask helper to process this molecule
            try:
                conn.send(('SMI', smi))
            except (BrokenPipeError, EOFError, OSError):
                # Helper died unexpectedly -> restart and skip this smi (empty)
                restart_helper()
                out_q.put(('ITEM', smi, []))
                continue

            # Wait up to timeout for a reply
            t0 = time.time()
            got = False
            while True:
                if conn.poll(timeout=0.01):
                    try:
                        tag, frags = conn.recv()
                        if tag == 'OK':
                            out_q.put(('ITEM', smi, frags))
                            got = True
                        else:
                            out_q.put(('ITEM', smi, []))
                            got = True
                    except (EOFError, OSError):
                        # Helper crashed during recv
                        out_q.put(('ITEM', smi, []))
                        got = True
                    break
                if (time.time() - t0) >= frag_timeout_sec:
                    # Timeout -> restart helper, skip this molecule
                    restart_helper()
                    out_q.put(('ITEM', smi, []))
                    got = True
                    break
            if not got:
                out_q.put(('ITEM', smi, []))
    except Exception as e:
        out_q.put(('ITEM', f'__EXC__:{repr(e)}', []))
    finally:
        # Gracefully stop helper
        try:
            conn.send(('STOP', None))
        except Exception:
            pass
        try:
            helper_proc.join(timeout=0.2)
        except Exception:
            pass
        if helper_proc.is_alive():
            try:
                helper_proc.terminate()
            except Exception:
                pass
            helper_proc.join(timeout=0.5)
        try:
            conn.close()
        except Exception:
            pass
        out_q.put(('DONE', None, None))


def _chunk_iter(seq: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield consecutive slices containing at most ``batch_size`` molecules."""
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def run_fragmentation_streaming(
    canon_list: List[str],
    params: dict,
    *,
    batch_size: int,
    q_maxsize: int,
    n_workers: int,
    frag_timeout_sec: int,
) -> Iterable[Tuple[str, List[str]]]:
    """Yield fragment results in canonical input order from bounded chunk workers.

    Parameters
    ----------
    canon_list : list of str
        Unique canonical molecules, normally from ``canonicalize_parallel``.
    params : dict
        ``min_heavy_atoms`` and ``max_frags_per_mol`` fragmentation settings.
    batch_size, q_maxsize, n_workers : int
        Molecules per chunk, queue message capacity, and active worker limit.
    frag_timeout_sec : int
        Maximum wall time in seconds for one helper request.

    Yields
    ------
    tuple of (str, list of str)
        Input molecule and its canonical capped fragment SMILES. Out-of-order
        results are buffered until earlier rows arrive; timeouts keep an empty row.

    Raises
    ------
    RuntimeError
        If a worker reports an exception, exits without completing, or loses rows.
    """
    start_method = "spawn" if sys.platform == "win32" else "fork"
    ctx = mp.get_context(start_method)
    out_q: mp.Queue = ctx.Queue(maxsize=q_maxsize)

    chunk_source = _chunk_iter(canon_list, batch_size)
    procs: List[mp.Process] = []
    active_workers = 0

    def _start_next() -> bool:
        nonlocal active_workers
        try:
            chunk = next(chunk_source)
        except StopIteration:
            return False
        p = ctx.Process(
            target=_frags_worker_stream,
            args=(chunk, out_q, params, frag_timeout_sec),
        )
        p.daemon = False
        p.start()
        procs.append(p)
        active_workers += 1
        return True

    for _ in range(n_workers):
        if not _start_next():
            break

    pending = {}
    next_row = 0
    try:
        while active_workers > 0:
            try:
                tag, smi, frags = out_q.get(timeout=1.0)
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    raise RuntimeError('Fragment workers exited before completing the input.')
                continue
            if tag == 'ITEM':
                if smi.startswith('__EXC__:'):
                    raise RuntimeError(smi)
                pending[smi] = frags
                while next_row < len(canon_list) and canon_list[next_row] in pending:
                    expected = canon_list[next_row]
                    yield expected, pending.pop(expected)
                    next_row += 1
            elif tag == 'DONE':
                active_workers -= 1
                # Reap finished procs
                for p in list(procs):
                    if not p.is_alive():
                        p.join(timeout=0.1)
                        procs.remove(p)
            # Bound out-of-order buffering if an earlier chunk contains slow molecules.
            while active_workers < n_workers and len(pending) < batch_size*n_workers:
                if not _start_next():
                    break
        if next_row != len(canon_list):
            raise RuntimeError('Fragmentation lost rows; refusing to emit misaligned data.')
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=1.0)


# ------------------------ main ------------------------

def parse_args() -> argparse.Namespace:
    """Parse input/output paths, fragmentation filters, worker limits, and timeouts."""
    ap = argparse.ArgumentParser("Extract BRICS fragments and build fragment vocab")
    ap.add_argument("--train-smiles", type=str, default=None,
                    help="Path to training SMILES (one per line). Preferred input.")
    ap.add_argument("--train-selfies", type=str, default=None,
                    help="Path to training SELFIES (one per line). Will be decoded to SMILES.")
    ap.add_argument("--out-jsonl", type=str, required=True,
                    help="Output JSONL with per-molecule fragments.")
    ap.add_argument("--out-vocab", type=str, required=True,
                    help="Output fragment vocabulary (SELFIES), sorted by frequency.")
    ap.add_argument("--out-freq", type=str, default=None,
                    help="Optional: CSV with fragment SELFIES and counts.")
    ap.add_argument("--min-heavy-atoms", type=int, default=5,
                    help="Minimum heavy atoms per fragment after capping.")
    ap.add_argument("--max-frags-per-mol", type=int, default=6,
                    help="Max fragments to keep per molecule (largest first).")
    ap.add_argument("--batch-size", type=int, default=400,
                    help="Molecules per worker chunk.")
    ap.add_argument("--queue-mult", type=int, default=4,
                    help="Queue capacity multiplier: q_maxsize = queue-mult * batch-size.")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="Optional cap for workers; default = CPU-3.")
    ap.add_argument("--frag-timeout-sec", type=int, default=60,
                    help="Wall time per molecule before helper restart.")
    ap.add_argument("--max-heavy-atoms-mol", type=int, default=None,
                    help="Skip molecules with heavy atoms > this (quick pre-filter).")
    return ap.parse_args()


def _ensure_parent_dir(path: str):
    """Create a file path's parent directories when the path has a parent."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _too_large_molecule(smi: str, max_heavy: Optional[int]) -> bool:
    """Return whether an active heavy-atom filter rejects a large/invalid molecule."""
    if not max_heavy:
        return False
    try:
        m = Chem.MolFromSmiles(smi, sanitize=False)
        return (m is None) or (m.GetNumHeavyAtoms() > max_heavy)
    except Exception:
        return True


def main():
    """Write ordered fragment JSONL, a frequency-ordered vocabulary, and optional CSV."""
    args = parse_args()

    if not args.train_smiles and not args.train_selfies:
        raise SystemExit("Provide --train-smiles or --train-selfies.")

    # Load input molecules as SMILES
    if args.train_smiles and os.path.exists(args.train_smiles):
        raw_smiles = read_smiles(args.train_smiles)
    elif args.train_selfies and os.path.exists(args.train_selfies):
        raw_smiles = read_selfies_decode_to_smiles(args.train_selfies)
    else:
        raise SystemExit("Input path not found.")

    # Optional quick pre-filter for very large molecules
    if args.max_heavy_atoms_mol:
        filtered = []
        skipped = 0
        for s in raw_smiles:
            if _too_large_molecule(s, args.max_heavy_atoms_mol):
                skipped += 1
            else:
                filtered.append(s)
        if skipped:
            print(f"Pre-filter skipped {skipped} large/heavy molecules.")
        raw_smiles = filtered

    # Canonicalize & deduplicate
    canon_list: List[str] = canonicalize_parallel(raw_smiles, chunksize=2000, workers=args.max_workers)

    # Prepare outputs
    _ensure_parent_dir(args.out_jsonl)
    _ensure_parent_dir(args.out_vocab)
    if args.out_freq:
        _ensure_parent_dir(args.out_freq)

    frag_counter: Counter = Counter()
    n_with_frags = 0

    # Parallel fragmentation settings
    cpu_total = os.cpu_count() or 4
    default_workers = max(1, cpu_total - 3)
    n_workers = max(1, min(args.max_workers or default_workers, default_workers))
    batch_size = max(80, args.batch_size)
    q_maxsize = max(2 * batch_size, args.queue_mult * batch_size)

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        with tqdm(total=len(canon_list), desc='BRICS fragmenting', unit='mol') as pbar:
            st_total = time.time()

            params = {
                'min_heavy_atoms': args.min_heavy_atoms,
                'max_frags_per_mol': args.max_frags_per_mol,
            }

            for (smi, frags_smiles) in run_fragmentation_streaming(
                canon_list,
                params,
                batch_size=batch_size,
                q_maxsize=q_maxsize,
                n_workers=n_workers,
                frag_timeout_sec=args.frag_timeout_sec,
            ):
                if smi.startswith('__EXC__:'):
                    pbar.update(0)
                    continue

                pbar.update(1)

                if not frags_smiles:
                    entry = {
                        "smiles": smi,
                        "selfies": smiles_to_selfies(smi) or "",
                        "frag_smiles": [],
                        "frag_selfies": []
                    }
                    fout.write(json.dumps(entry) + "\n")
                    continue

                n_with_frags += 1
                frag_selfies: List[str] = []
                for fsmi in frags_smiles:
                    sel = smiles_to_selfies(fsmi)
                    if sel:
                        frag_selfies.append(sel)
                        frag_counter[sel] += 1

                entry = {
                    "smiles": smi,
                    "selfies": smiles_to_selfies(smi) or "",
                    "frag_smiles": frags_smiles,
                    "frag_selfies": frag_selfies
                }
                fout.write(json.dumps(entry) + "\n")

            elapsed_total = time.time() - st_total
            print(f"\n<+> Total Time: {elapsed_total:.2f}s")

    # Build vocab sorted by frequency desc
    vocab_tokens = [frag for frag, _cnt in frag_counter.most_common()]
    with open(args.out_vocab, "w", encoding="utf-8") as f:
        for t in vocab_tokens:
            f.write(t + "\n")

    if args.out_freq:
        with open(args.out_freq, "w", encoding="utf-8") as f:
            f.write("fragment_selfies,count\n")
            for frag, cnt in frag_counter.most_common():
                f.write(f"{frag},{cnt}\n")

    total = len(canon_list)
    print("\n=== Fragment extraction summary ===")
    print(f"Input molecules:       {len(raw_smiles)}")
    print(f"Canonical unique:      {total}")
    print(f"With >=1 fragment:     {n_with_frags} ({100.0 * n_with_frags / max(1, total):.2f}%)")
    print(f"Fragment vocab size:   {len(vocab_tokens)}")
    print(f"JSONL written to:      {args.out_jsonl}")
    print(f"Vocab written to:      {args.out_vocab}")
    if args.out_freq:
        print(f"Freq CSV written to:   {args.out_freq}")


if __name__ == "__main__":
    # Required on Windows
    mp.freeze_support()
    main()
