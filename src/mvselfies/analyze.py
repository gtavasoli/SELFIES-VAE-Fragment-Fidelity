"""Describe generated molecules with RDKit properties and diversity summaries.

The CLI writes per-molecule properties to CSV and prints aggregate statistics.
``sa_proxy`` is a simple descriptor heuristic, not an Ertl SA score.
"""
import os, argparse, random, statistics
from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors as rdmd
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
import math
import csv

def read_smiles(path:str) -> List[str]:
    """Read nonempty UTF-8 SMILES lines without canonicalizing or deduplicating."""
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def canonical_smiles(s: str):
    """Return ``(mol, canonical_isomeric_smiles)`` or ``(None, None)`` on failure."""
    try:
        m = Chem.MolFromSmiles(s, sanitize=True)
        if not m: return None, None
        can = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
        return m, can
    except Exception:
        return None, None

def props(m: Chem.Mol) -> Tuple[float,float,float,float,int,int,int]:
    """Calculate the property tuple written to each molecular CSV row.

    Parameters
    ----------
    m : rdkit.Chem.Mol
        Parsed, sanitized molecule.

    Returns
    -------
    tuple
        Molecular weight, Crippen logP, QED, SA proxy, ring count,
        hydrogen-bond donor count, and hydrogen-bond acceptor count, in order.

    Notes
    -----
    The SA proxy is ``0.004*MW + 0.5*rings + 0.1*(HBA+HBD)``. It is reported as
    a heuristic and is not a calibrated synthetic-accessibility score.
    """
    mw   = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    qed  = QED.qed(m)
    rings = rdmd.CalcNumRings(m)
    hbd   = rdmd.CalcNumHBD(m)
    hba   = rdmd.CalcNumHBA(m)
    # simple SA proxy (lower is easier): use MW, rings, HBA/HBD as rough heuristic
    sa = 0.004*mw + 0.5*rings + 0.1*(hba+hbd)
    return mw, logp, qed, sa, rings, hbd, hba

def murcko(smiles: str) -> str:
    """Return canonical Murcko scaffold SMILES, or an empty string if unavailable."""
    m = Chem.MolFromSmiles(smiles)
    if not m: return ""
    scaff = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(scaff, canonical=True) if scaff else ""

def ecfp4(m: Chem.Mol):
    """Return a 2048-bit radius-two Morgan fingerprint for Tanimoto diversity."""
    return AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)

def summarize(vals: List[float]) -> dict:
    """Return mean, population standard deviation, and indexed p10/p50/p90.

    Percentiles use the recorded sorted-index rule, without interpolation.
    Empty input returns an empty dictionary.
    """
    if not vals: return {}
    vals_sorted = sorted(vals)
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals)>1 else 0.0,
        "p10": vals_sorted[max(0, int(0.10*len(vals))-1)],
        "p50": vals_sorted[max(0, int(0.50*len(vals))-1)],
        "p90": vals_sorted[max(0, int(0.90*len(vals))-1)],
    }

def main():
    """Write generated-molecule properties and print diversity/reference summaries."""
    ap = argparse.ArgumentParser("Analyze generated SMILES quality & diversity")
    ap.add_argument("--gen", required=True, help="generated canonical SMILES (from your previous step)")
    ap.add_argument("--train", required=False, help="training canonical SMILES (to compare dists; optional)")
    ap.add_argument("--sample-for-diversity", type=int, default=1500, help="subset size for internal diversity")
    ap.add_argument("--out-csv", required=True, help="path to save per-molecule props CSV")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)

    gen_smiles = read_smiles(args.gen)
    train_smiles = read_smiles(args.train) if args.train and os.path.exists(args.train) else []

    # Compute properties for generated set
    from tqdm import tqdm

    rows, mw_list, logp_list, qed_list, sa_list = [], [], [], [], []
    rings_list, hbd_list, hba_list, scaffolds = [], [], [], []

    for smi in tqdm(gen_smiles, total=len(gen_smiles), desc="Reading SMILES", unit="mol", dynamic_ncols=True):
        m, can = canonical_smiles(smi)
        if not m:
            continue
        mw, logp, qed, sa, rings, hbd, hba = props(m)
        rows.append([can, mw, logp, qed, sa, rings, hbd, hba, murcko(can)])
        mw_list.append(mw); logp_list.append(logp); qed_list.append(qed); sa_list.append(sa)
        rings_list.append(rings); hbd_list.append(hbd); hba_list.append(hba)
        scaffolds.append(rows[-1][-1])

    # Save CSV
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["smiles","mw","logp","qed","sa_proxy","rings","hbd","hba","murcko_scaffold"])
        w.writerows(rows)

    # Diversity metrics
    unique_scaff = len(set([s for s in scaffolds if s]))
    scaffold_div = unique_scaff / max(1, len(scaffolds))

    # Internal diversity via Tanimoto on a subset
    # (compute on fingerprints in a small subset to keep runtime modest)
    from rdkit import DataStructs
    subset_smiles = random.sample([r[0] for r in rows], min(args.sample_for_diversity, len(rows)))
    fps = [ecfp4(Chem.MolFromSmiles(s)) for s in subset_smiles]
    # pairwise mean distance = 1 - mean similarity
    sims = []
    for i in range(len(fps)):
        fi = fps[i]
        for j in range(i+1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fi, fps[j]))
    internal_div = 1.0 - (sum(sims)/len(sims)) if sims else 0.0

    # Summaries
    mw_stats   = summarize(mw_list)
    logp_stats = summarize(logp_list)
    qed_stats  = summarize(qed_list)
    sa_stats   = summarize(sa_list)
    rings_stats= summarize(rings_list)

    print("\n=== Generated set quality & diversity ===")
    print(f"n_valid: {len(rows)}")
    print(f"Scaffold uniqueness: {unique_scaff}/{len(scaffolds)} = {scaffold_div*100:.2f}%")
    print(f"Internal diversity (ECFP4 mean distance on {len(fps)} subset): {internal_div:.3f}")
    print("MW   :", mw_stats)
    print("logP :", logp_stats)
    print("QED  :", qed_stats)
    print("SA*  :", sa_stats, "  (*proxy unless you use sascorer)")
    print("Rings:", rings_stats)

    # Optional: compare distributions to training (rough mean/median)
    if train_smiles:
        tmw, tlogp, tqed, tsa, trings = [], [], [], [], []
        for smi in tqdm(
            train_smiles,
            total=len(train_smiles),
            desc="Analyzing training molecules",
            unit="mol",
            dynamic_ncols=True
        ):
            m, _ = canonical_smiles(smi)
            if not m:
                continue
            mw, logp, qed, sa, rings, _, _ = props(m)
            tmw.append(mw); tlogp.append(logp); tqed.append(qed); tsa.append(sa); trings.append(rings)
        def _cmp(name, gen, trn):
            if not trn: return
            gs, ts = summarize(gen), summarize(trn)
            print(f"{name} — gen mean {gs.get('mean',float('nan')):.3f} vs train {ts.get('mean',float('nan')):.3f}, "
                  f"gen p10/50/90 {gs.get('p10','?')}/{gs.get('p50','?')}/{gs.get('p90','?')} "
                  f"vs train {ts.get('p10','?')}/{ts.get('p50','?')}/{ts.get('p90','?')}")
        print("\n--- Gen vs Train (rough) ---")
        _cmp("MW", mw_list, tmw)
        _cmp("logP", logp_list, tlogp)
        _cmp("QED", qed_list, tqed)
        _cmp("SA*", sa_list, tsa)
        _cmp("Rings", rings_list, trings)

if __name__ == "__main__":
    main()
