"""Train Contrastive-VAE and Rare-Aware-VAE with fragment-set supervision.

The molecular backbone is shared with Baseline-VAE. Optional objectives include
one-way InfoNCE, a fragment bag-of-words BCE loss, and attention-entropy reward.
KL weighting is warmed up; fragment pooling and dropout follow recorded settings.

Notes
-----
InfoNCE receives detached molecular means by default. With attention pooling,
the undetached mean remains the attention query and provides an indirect gradient
path. ``--symmetric-info`` removes the detach; it does not add reverse-direction
InfoNCE. Checkpoints save the molecular VAE, not the auxiliary fragment modules.
"""

import os, math, random, argparse, json, datetime, contextlib, csv
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# The molecular VAE returns its posterior parameters for fragment supervision.
from .model import BaselineVAE
from .tokenizer import SelfiesTokenizer

# ---------------- AMP shim ----------------
try:
    from torch.amp import autocast, GradScaler
    AMP_KW = dict(device_type="cuda")
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    AMP_KW = dict(enabled=torch.cuda.is_available())
USE_AMP = torch.cuda.is_available()

# ---------------- Logger ----------------
def _make_logger(log_path: Optional[str] = None):
    """Return a timestamped console logger with optional append-only file output."""
    f = None
    if log_path:
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        f = open(log_path, "a", encoding="utf-8")
    def _log(*msg, sep=" "):
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] " + sep.join(map(str, msg))
        print(line)
        if f:
            f.write(line + "\n"); f.flush()
    return _log

# ---------------- Numerics helpers ----------------
def safe_exp(x: float, cap: float = 50.0) -> float:
    """Numerically safe exp for perplexity logging."""
    try:
        if not math.isfinite(x): return float("inf")
        return math.exp(min(max(0.0, x), cap))
    except Exception:
        return float("inf")

# ---------------- Tokenizer ----------------
# ---------------- CSV helpers ----------------
def _norm(s: str) -> str:
    """Normalize frequency-CSV header names for tolerant column matching."""
    return s.strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def _read_csv_rows(csv_path: str):
    """Read a nonempty CSV with a sniffed dialect; raise ValueError if empty."""
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    return rows

# ---------------- BoW pos_weight loader ----------------
def load_pos_weight_from_csv(csv_path: str, frag_vocab_path: str, clamp_max: float = 50.0) -> torch.Tensor:
    """Compute vocabulary-aligned positive-class weights for fragment BoW BCE.

    Parameters
    ----------
    csv_path : str
        Fragment/count CSV. Recognized header aliases are detected; otherwise
        the first two columns are interpreted as fragment and count.
    frag_vocab_path : str
        Fragment vocabulary in output-class order, using the CSV's representation.
    clamp_max : float or None, default=50.0
        Upper bound for weights, or None to leave them unclipped.

    Returns
    -------
    torch.Tensor
        CPU float32 vector of shape (V,). Each weight is ``(N-c)/max(1,c)``,
        where N includes all parseable CSV counts. An all-zero total gives ones.
    """
    with open(frag_vocab_path, "r", encoding="utf-8") as f:
        vocab = [ln.strip() for ln in f if ln.strip()]
    idx = {t:i for i,t in enumerate(vocab)}
    counts = [0]*len(vocab); N=0

    rows = _read_csv_rows(csv_path)
    frag_names  = {"fragment","frag","token","piece","brics","smarts","smiles"}
    count_names = {"count","freq","frequency","n","num"}
    header = [_norm(x) for x in rows[0]]
    frag_col = count_col = None
    for j,h in enumerate(header):
        if h in frag_names and frag_col is None: frag_col=j
        if h in count_names and count_col is None: count_col=j
    data = rows[1:] if (frag_col is not None and count_col is not None) else rows
    if frag_col is None or count_col is None:
        frag_col, count_col = 0, 1

    for r in data:
        if len(r) <= max(frag_col,count_col): continue
        frag = r[frag_col].strip()
        if not frag: continue
        try:
            c = int(float(r[count_col]))
        except Exception:
            continue
        N += c
        if frag in idx:
            counts[idx[frag]] += c

    if N == 0:
        pw = [1.0]*len(vocab)
    else:
        pw=[]
        for c in counts:
            val = (N - c) / max(1, c)
            if clamp_max is not None:
                val = min(val, clamp_max)
            pw.append(val)
    return torch.tensor(pw, dtype=torch.float32)

# ---------------- Rare-aware dropout vectors ----------------
def load_dropout_vectors_from_csv(
    csv_path: str,
    frag_vocab_path: str,
    alpha: float = 0.7,
    rare_quantile: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build frequency-rank dropout factors and a rare-fragment membership mask.

    Parameters
    ----------
    csv_path, frag_vocab_path : str
        Fragment/count CSV and matching ordered fragment vocabulary.
    alpha : float, default=0.7
        Exponent applied to ranks normalized from zero (most frequent) to one
        (least frequent). The trainer later scales these factors by base dropout.
    rare_quantile : float, default=0.2
        Select the first ``int(rare_quantile*(V-1))+1`` entries in ascending count
        order as rare. Ties follow vocabulary order.

    Returns
    -------
    p_drop_vec : torch.Tensor
        CPU float32 factors of shape (V,); larger rarity ranks get larger factors.
    rare_mask : torch.Tensor
        CPU boolean vector of shape (V,) marking the rare subset.
    """
    with open(frag_vocab_path, "r", encoding="utf-8") as f:
        vocab = [ln.strip() for ln in f if ln.strip()]
    V = len(vocab)
    idx = {t:i for i,t in enumerate(vocab)}
    counts = [0]*V

    rows = _read_csv_rows(csv_path)
    frag_names  = {"fragment","frag","token","piece","brics","smarts","smiles"}
    count_names = {"count","freq","frequency","n","num"}
    header = [_norm(x) for x in rows[0]]
    frag_col = count_col = None
    for j,h in enumerate(header):
        if h in frag_names and frag_col is None: frag_col=j
        if h in count_names and count_col is None: count_col=j
    data = rows[1:] if (frag_col is not None and count_col is not None) else rows
    if frag_col is None or count_col is None:
        frag_col, count_col = 0, 1

    for r in data:
        if len(r) <= max(frag_col,count_col): continue
        frag = r[frag_col].strip()
        if not frag: continue
        try: c = int(float(r[count_col]))
        except Exception: continue
        if frag in idx: counts[idx[frag]] = c

    order = sorted(range(V), key=lambda i: counts[i], reverse=True)
    rank = [0]*V
    for pos, i in enumerate(order):
        rank[i] = pos

    rnorm = [pos/(V-1) if V>1 else 0.0 for pos in rank]
    p_drop = [(rn ** alpha) for rn in rnorm]

    thr_index = int(rare_quantile * (V-1))
    sorted_by_rare = sorted(range(V), key=lambda i: counts[i])  # asc
    rare_set = set(sorted_by_rare[:thr_index+1])

    p_drop_vec = torch.tensor(p_drop, dtype=torch.float32)
    rare_mask  = torch.tensor([i in rare_set for i in range(V)], dtype=torch.bool)
    return p_drop_vec, rare_mask

# ---------------- Dataset ----------------
class MolWithFragSetDataset(Dataset):
    """Pair molecular SELFIES with fragment sets and optional sparse BoW labels.

    Parameters
    ----------
    selfies_path, frags_jsonl : str
        Molecular SELFIES file and fragment JSONL with matching row count and
        molecule identities in ``selfies`` fields.
    tok : SelfiesTokenizer
        Shared molecular/fragment sequence tokenizer.
    max_len, max_frags_per_mol, max_frag_len : int
        Molecular sequence length, retained fragment limit, and fragment length.
    frag_vocab_path : str, optional
        Ordered whole-fragment vocabulary for dropout indices; unknowns use -1.
    frag_labels_pkl : str, optional
        Per-row lists of vocabulary indices. Multi-hot vectors are expanded only
        when an item is requested.
    frag_vocab_size : int, optional
        BoW class count, required when a label pickle is supplied.
    legacy_fragment_order : bool, default=False
        Permit historical positional pairing despite identity mismatches and
        emit a warning. Row-count checks still apply.

    Raises
    ------
    ValueError
        If identities differ without the override, or label dimensionality is missing.
    AssertionError
        If molecule, fragment, or supplied label row counts differ.
    """
    def __init__(self, selfies_path: str, frags_jsonl: str, tok: SelfiesTokenizer,
                 max_len: int, max_frags_per_mol: int = 6, max_frag_len: int = 60,
                 frag_vocab_path: Optional[str] = None,
                 frag_labels_pkl: Optional[str] = None,
                 frag_vocab_size: Optional[int] = None,
                 legacy_fragment_order: bool = False):
        import pickle
        with open(selfies_path,"r",encoding="utf-8") as f:
            self.mols=[ln.strip() for ln in f if ln.strip()]
        self.frag_sets=[]
        mismatched_rows = 0
        with open(frags_jsonl,"r",encoding="utf-8") as f:
            for row, ln in enumerate(f):
                d=json.loads(ln)
                if row < len(self.mols) and d.get('selfies') != self.mols[row]:
                    mismatched_rows += 1
                    if not legacy_fragment_order:
                        raise ValueError(
                            f'Fragment row {row+1} does not match the molecule file. '
                            'Regenerate aligned fragments for new experiments. '
                            '--legacy-fragment-order reproduces the archived positional pairing.'
                        )
                frs=d.get("frag_selfies",[]) or []
                self.frag_sets.append(frs)
        assert len(self.mols)==len(self.frag_sets), "molecules and frags JSONL must have same length/order"
        if mismatched_rows:
            import warnings
            warnings.warn(f'Reproducing historical positional pairing: {mismatched_rows} mismatched fragment rows.')
        self.tok=tok
        self.max_len=max_len
        self.max_frags=max_frags_per_mol
        self.max_frag_len=max_frag_len

        self.frag_to_idx = None
        if frag_vocab_path:
            with open(frag_vocab_path,"r",encoding="utf-8") as f:
                vocab = [ln.strip() for ln in f if ln.strip()]
            self.frag_to_idx = {t:i for i,t in enumerate(vocab)}

        self.labels=None
        self.frag_vocab_size=frag_vocab_size
        if frag_labels_pkl:
            with open(frag_labels_pkl,"rb") as f:
                raw = pickle.load(f)  # list[list[int]]
            assert len(raw)==len(self.mols), "labels size must match molecules"
            if frag_vocab_size is None:
                raise ValueError("frag_vocab_size must be provided when using frag_labels_pkl")
            # Expand only the requested row; corpus-wide dense labels need gigabytes.
            self.labels=raw

    def __len__(self): return len(self.mols)

    def _encode_frag_set(self, frags: List[str]):
        """Return fragment input indices, padding masks, and vocabulary indices.

        Outputs have shapes (F, T_fragment), (F, T_fragment), and (F,), with True
        marking padding and -1 marking unknown whole fragments. Empty sets get
        one fully padded row so later batching always has a fragment dimension.
        """
        selected = frags[:self.max_frags] if frags else []
        xs, pads, idxs = [], [], []

        if not selected:
            ids_in = [self.tok.pad_id] * self.max_frag_len
            xs.append(torch.tensor(ids_in, dtype=torch.long))
            pads.append(torch.ones(self.max_frag_len, dtype=torch.bool))
            idxs.append(-1)
        else:
            for fs in selected:
                if not fs:
                    ids_in = [self.tok.pad_id] * self.max_frag_len
                    pad_row = torch.ones(self.max_frag_len, dtype=torch.bool)
                    frag_i = -1
                else:
                    ids_in, _, _ = self.tok.encode(fs, self.max_frag_len)
                    pad_row = torch.tensor([tid == self.tok.pad_id for tid in ids_in],
                                           dtype=torch.bool)
                    frag_i = self.frag_to_idx.get(fs, -1) if self.frag_to_idx is not None else -1
                xs.append(torch.tensor(ids_in, dtype=torch.long))
                pads.append(pad_row)
                idxs.append(frag_i)

        x = torch.stack(xs, dim=0)                 # (F, T)
        pad = torch.stack(pads, dim=0)             # (F, T)  True==pad
        frag_idx = torch.tensor(idxs, dtype=torch.long)  # (F,)
        return x, pad, frag_idx

    def __getitem__(self, idx):
        """Return molecule tensors, fragment tensors/indices, and optional BoW.

        The tuple is ``(x_in, x_out, pad, frag_x, frag_pad, frag_idx, bow)``.
        Molecular tensors have shape (T,); fragment tensors have (F, T_fragment)
        and indices have (F,). BoW is a float32 vector (V,) or None.
        """
        mol=self.mols[idx]
        frs=self.frag_sets[idx]
        x_in, x_out, pad = self.tok.encode(mol, self.max_len)
        frag_x, frag_pad, frag_idx = self._encode_frag_set(frs)
        bow = None
        if self.labels is not None:
            bow = torch.zeros(self.frag_vocab_size, dtype=torch.float32)
            if self.labels[idx]:
                bow[self.labels[idx]] = 1.0
        return (
            torch.tensor(x_in,  dtype=torch.long),
            torch.tensor(x_out, dtype=torch.long),
            torch.tensor(pad,   dtype=torch.bool),
            frag_x, frag_pad, frag_idx,
            bow
        )

def collate(batch):
    """Stack molecular inputs and pad variable fragment counts to the batch maximum.

    Returns the dataset's seven fields with an added batch dimension: molecular
    tensors (B, T), fragment tensors (B, F, T_fragment), indices (B, F), and BoW
    labels (B, V) or None. Added fragment rows use PAD=0, True masks, and index -1.
    """
    x_in  = torch.stack([b[0] for b in batch], 0)
    x_out = torch.stack([b[1] for b in batch], 0)
    pad   = torch.stack([b[2] for b in batch], 0)
    frag_list  = [b[3] for b in batch]
    fpad_list  = [b[4] for b in batch]
    fidx_list  = [b[5] for b in batch]
    bows = None
    if batch[0][6] is not None:
        bows = torch.stack([b[6] for b in batch], 0)  # (B,V)

    maxF = max(x.size(0) for x in frag_list)
    T    = frag_list[0].size(1)
    batch_frag, batch_fpad, batch_fidx = [], [], []
    for x, m, fi in zip(frag_list, fpad_list, fidx_list):
        F = x.size(0)
        if F < maxF:
            pad_x = torch.full((maxF - F, T), 0, dtype=torch.long)
            pad_m = torch.ones((maxF - F, T), dtype=torch.bool)
            pad_i = torch.full((maxF - F,), -1, dtype=torch.long)
            x = torch.cat([x, pad_x], 0)
            m = torch.cat([m, pad_m], 0)
            fi = torch.cat([fi, pad_i], 0)
        batch_frag.append(x)
        batch_fpad.append(m)
        batch_fidx.append(fi)
    frag_x  = torch.stack(batch_frag, 0)   # (B, F, T)
    frag_pad= torch.stack(batch_fpad, 0)   # (B, F, T)
    frag_idx= torch.stack(batch_fidx, 0)   # (B, F)
    return x_in, x_out, pad, frag_x, frag_pad, frag_idx, bows

# ---------------- Fragment encoder with attention pooling ----------------
class FragEncoder(nn.Module):
    """Encode fragment sequences with an LSTM, then mean- or attention-pool sets.

    Parameters
    ----------
    vocab_size, pad_id : int
        Shared sequence-vocabulary size and padding index.
    emb, hid, z_dim : int
        Token embedding, recurrent hidden, and projected fragment dimensions.
    layers : int, default=1
        Number of unidirectional LSTM layers.
    dropout : float, default=0.1
        Inter-layer recurrent dropout, active only with more than one layer.
    attn_dim : int, optional
        Query/key projection dimension; defaults to ``z_dim``.
    """
    def __init__(self, vocab_size:int, pad_id:int, emb:int=256, hid:int=512, z_dim:int=64,
                 layers:int=1, dropout:float=0.1, attn_dim: Optional[int]=None):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb, hid, num_layers=layers, batch_first=True,
                            bidirectional=False, dropout=dropout if layers>1 else 0.0)
        self.proj = nn.Sequential(
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, z_dim),
        )
        d = attn_dim or z_dim
        self.Wq = nn.Linear(z_dim, d, bias=False)
        self.Wk = nn.Linear(z_dim, d, bias=False)
        self.Wv = nn.Linear(z_dim, z_dim, bias=False)
        self.scaler = 1.0 / math.sqrt(d)

    def encode_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Project final LSTM states from (N, T_fragment) indices to (N, Z).

        Sequences run at their padded length without packing. Set-level masking
        of fully padded rows is applied by ``forward`` after encoding.
        """
        e = self.emb(x)               # (N, T, E)
        out, (h, _) = self.lstm(e)    # h: (layers, N, H)
        h_last = h[-1]                # (N, H)
        return self.proj(h_last)      # (N, z)

    def forward(self,
                frag_x: torch.Tensor, frag_pad: torch.Tensor,
                drop_frag_p: float = 0.0, training: bool = True,
                frag_idx: Optional[torch.Tensor] = None,           # (B,F)
                p_drop_vec: Optional[torch.Tensor] = None,         # (V,)
                keep_at_least_one_rare: bool = True,
                rare_mask_vec: Optional[torch.Tensor] = None,      # (V,)
                attn_query: Optional[torch.Tensor] = None,         # mu, (B,z)
                use_attn: bool = True,
                attn_temp: float = 1.0,
                return_attn: bool = False):
        """Apply optional fragment dropout and pool the surviving row embeddings.

        Parameters
        ----------
        frag_x, frag_pad : torch.Tensor
            Integer fragment indices and boolean padding masks, both (B, F, T).
        drop_frag_p : float, default=0.0
            Scale for per-vocabulary dropout factors; probabilities are clamped
            to [0, 1]. Dropout needs ``training``, indices, and factors supplied.
        training : bool, default=True
            Enable the fragment-dropout branch, independently of module mode.
        frag_idx : torch.Tensor, optional
            Whole-fragment indices (B, F), with -1 for unknowns.
        p_drop_vec, rare_mask_vec : torch.Tensor, optional
            Vocabulary-aligned dropout factors and boolean rare-membership mask.
        keep_at_least_one_rare : bool, default=True
            Restore a rare row if all present rare fragments were dropped.
            At least one valid row is also restored for nonempty sets.
        attn_query : torch.Tensor, optional
            Molecular means (B, Z) used as differentiable attention queries.
        use_attn : bool, default=True
            Use attention when a query is supplied; otherwise use masked means.
        attn_temp : float, default=1.0
            Positive divisor for attention scores.
        return_attn : bool, default=False
            Include attention diagnostics when the attention branch is active.

        Returns
        -------
        pooled : torch.Tensor
            Set embedding of shape (B, Z), zero for fully padded sets.
        attention, valid_rows : torch.Tensor, optional
            When attention is active and requested, return a three-item tuple
            with weights (B, F) and surviving-row counts (B,). Mean pooling
            returns only ``pooled`` even when ``return_attn=True``.
        """
        B, F, T = frag_x.shape
        x = frag_x.clone()
        m = frag_pad.clone()  # (B,F,T)

        # ------- rare-aware dropout (row-wise) -------
        if training and drop_frag_p > 1e-6 and frag_idx is not None and p_drop_vec is not None:
            full_pad = m.all(dim=2)             # (B,F)
            valid = ~full_pad
            p_rows = torch.zeros(B, F, device=x.device)
            has_idx = (frag_idx >= 0)
            p_rows[has_idx] = p_drop_vec[frag_idx[has_idx]]
            p_rows = (p_rows * drop_frag_p).clamp(0, 1)
            drop = (torch.rand(B, F, device=x.device) < p_rows) & valid

            keep_count = (valid & ~drop).sum(dim=1)
            need_fix = keep_count == 0
            if need_fix.any():
                fix_idx = need_fix.nonzero(as_tuple=False).squeeze(1)
                for b in fix_idx.tolist():
                    v_rows = (valid[b]).nonzero(as_tuple=False).squeeze(1)
                    if v_rows.numel() == 0:  # no valid fragment rows anyway
                        continue
                    j_choice = torch.randint(low=0, high=v_rows.numel(), size=()).item()
                    j = v_rows[j_choice].item()
                    drop[b, j] = False

            if keep_at_least_one_rare and rare_mask_vec is not None:
                rare_rows = torch.zeros(B, F, dtype=torch.bool, device=x.device)
                has_idx = (frag_idx >= 0)
                rare_rows[has_idx] = rare_mask_vec[frag_idx[has_idx]]
                rare_present = (rare_rows & valid).any(dim=1)
                need_keep_rare = rare_present & ~( (rare_rows & ~drop).any(dim=1) )
                if need_keep_rare.any():
                    for b in need_keep_rare.nonzero(as_tuple=False).squeeze(1).tolist():
                        rare_list = (rare_rows[b] & valid[b]).nonzero(as_tuple=False).squeeze(1)
                        if rare_list.numel() == 0:
                            continue
                        j_choice = torch.randint(low=0, high=rare_list.numel(), size=()).item()
                        j = rare_list[j_choice].item()
                        drop[b, j] = False

            if drop.any():
                x[drop] = self.pad_id
                m[drop] = True

        # ------- encode all rows, then pool -------
        flat = x.view(B*F, T)
        z = self.encode_seq(flat).view(B, F, -1)  # (B, F, z)
        full_pad_after = m.all(dim=2)             # (B,F)
        mask = ~full_pad_after                    # (B,F)

        if use_attn and (attn_query is not None):
            Q = self.Wq(attn_query).unsqueeze(1)          # (B,1,d)
            K = self.Wk(z)                                 # (B,F,d)
            V = self.Wv(z)                                 # (B,F,z)

            # Compute scores in fp32 to avoid fp16 underflow on large negatives
            scores = (Q.float() * K.float()).sum(dim=-1) * float(self.scaler)  # [B,F]
            if attn_temp and attn_temp > 0:
                scores = scores / float(attn_temp)

            NEG = -1e4  # safe for fp16
            scores = scores.masked_fill(~mask, NEG)

            w = torch.softmax(scores, dim=1)  # Normalize empty rows in FP32.
            w = w * mask
            den = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
            w = (w / den).to(z.dtype)

            pooled = (w.unsqueeze(-1) * V).sum(dim=1)  # [B,z]
            if return_attn:
                valid_rows = mask.sum(dim=1)  # (B,)
                return (pooled, w, valid_rows)
            else:
                return pooled
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
            pooled = (z * mask.unsqueeze(-1)).sum(dim=1) / denom.squeeze(1)
            return pooled

# ---------------- InfoNCE ----------------
def info_nce(z_q: torch.Tensor, z_k: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Compute one-way InfoNCE using paired rows and in-batch negatives.

    Parameters
    ----------
    z_q, z_k : torch.Tensor
        Corresponding query/key embeddings of shape (B, Z), normalized here.
    temperature : float, default=0.07
        Logit divisor, floored at 1e-8.

    Returns
    -------
    torch.Tensor
        Scalar mean cross-entropy with diagonal positive-pair targets. Inputs
        are not detached here; the training loop controls gradient flow.
    """
    z_q = F.normalize(z_q, dim=-1)
    z_k = F.normalize(z_k, dim=-1)
    logits = (z_q @ z_k.t()) / max(1e-8, temperature)  # (B,B)
    targets = torch.arange(z_q.size(0), device=z_q.device)
    return nn.CrossEntropyLoss()(logits, targets)

# ---------------- Utils ----------------
def set_seed(seed:int):
    """Seed Python, NumPy, and PyTorch RNGs and request deterministic algorithms.

    Unsupported deterministic operations emit warnings; numerical equivalence
    across library versions and hardware is not guaranteed.
    """
    random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def seed_worker(_):
    """Seed DataLoader worker Python/NumPy RNGs from PyTorch's worker seed."""
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws); random.seed(ws)

def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, args: argparse.Namespace):
    """Save the molecular VAE, optimizer, vocabulary, step, and CLI settings.

    Auxiliary fragment-encoder and BoW-head weights are omitted. These
    checkpoints support VAE inference but lack the full multiview training state.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "vocab": SelfiesTokenizer(args.vocab).id_to_tok,
                "opt_state": optimizer.state_dict(),
                "step": step,
                "args": {k:v for k,v in vars(args).items() if k!="log"}}, path)

# ---------------- CLI ----------------
def parse_args():
    """Parse multiview settings and attach a logger to the returned namespace."""
    p = argparse.ArgumentParser("Contrastive-VAE / Rare-Aware-VAE: InfoNCE + fragment dropout + attention pooling")
    p.add_argument("--data", required=True, type=str)
    p.add_argument("--vocab", required=True, type=str)
    p.add_argument("--max-len", required=True, type=int)
    p.add_argument("--frags-jsonl", required=True, type=str)
    p.add_argument("--legacy-fragment-order", action="store_true",
                   help="Reproduce archived positional pairing despite molecule/fragment identity mismatches.")

    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--emb", type=int, default=256)
    p.add_argument("--hid", type=int, default=512)
    p.add_argument("--z", type=int, default=64)
    p.add_argument("--enc-layers", type=int, default=2)
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)

    # InfoNCE
    p.add_argument("--gamma-info", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--gamma-info-warmup", type=int, default=0, help="linear warmup steps for gamma (0=off)")
    p.add_argument("--symmetric-info", action="store_true", help="if set, do NOT stop-grad on mu (default: stop-grad)")

    # KL scheduling
    p.add_argument("--beta-steps", type=int, default=100000)
    p.add_argument("--free-bits", type=float, default=0.0)

    # Fragments
    p.add_argument("--max-frags-per-mol", type=int, default=6)
    p.add_argument("--max-frag-len", type=int, default=60)

    # Dropout controls
    p.add_argument("--frag-dropout", type=float, default=0.30, help="base dropout prob")
    p.add_argument("--rare-aware-dropout", action="store_true", help="enable rare-aware scaling")
    p.add_argument("--frag-drop-alpha", type=float, default=0.7, help="power for rank^alpha")
    p.add_argument("--frag-rare-quantile", type=float, default=0.2, help="bottom quantile considered rare")

    # BoW (optional)
    p.add_argument("--lambda-frag-bow", type=float, default=0.0)
    p.add_argument("--frag-vocab", type=str, default=None)
    p.add_argument("--frag-labels-pkl", type=str, default=None)
    p.add_argument("--frag-freq", type=str, default=None)
    p.add_argument("--bow-clamp-max", type=float, default=50.0)

    # Attention pooling + regularization
    p.add_argument("--attn-pooling", action="store_true")
    p.add_argument("--attn-temp", type=float, default=1.0, help="softmax temperature (>1 = flatter)")
    p.add_argument("--attn-entropy", type=float, default=0.0, help="encourage flatter attention by subtracting lambda * H(w)")

    # Infra
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-file", type=str, default=None)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--save-interval", type=int, default=2000)

    # MLflow
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--mlflow-uri", type=str, default=None)
    p.add_argument("--mlflow-experiment", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--mlflow-tags", type=str, default=None)

    args = p.parse_args()
    args.log = _make_logger(args.log_file)
    return args

# ---------------- Train ----------------
def main():
    """Train the configured multiview variant and select checkpoints by validation CE.

    KL is linearly warmed up; optional free bits subtract a threshold from the
    scalar KL loss. Attention pooling limits InfoNCE to rows with at least two
    surviving fragments. The mean-pooling branch includes all batch rows.
    """
    args = parse_args()
    set_seed(args.seed)

    # MLflow
    ml = None
    if args.mlflow:
        try:
            import mlflow as ml
            if args.mlflow_uri: ml.set_tracking_uri(args.mlflow_uri)
            if args.mlflow_experiment: ml.set_experiment(args.mlflow_experiment)
            _ = ml.start_run(run_name=args.run_name)
            log_params = {k: v for k, v in vars(args).items()
                          if k not in ("log","mlflow","mlflow_uri","mlflow_experiment","run_name","mlflow_tags")}
            ml.log_params(log_params)
            if args.mlflow_tags:
                tags = dict(kv.split("=",1) for kv in args.mlflow_tags.split(",") if "=" in kv)
                ml.set_tags(tags)
        except Exception as e:
            print(f"[MLflow] Disabled due to error: {e}")
            ml = None

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.log(f"Using device: {device}")

    # Tokenizer & data
    tok = SelfiesTokenizer(args.vocab)

    frag_vocab_size = None
    if args.frag_vocab:
        with open(args.frag_vocab,"r",encoding="utf-8") as f:
            frag_vocab_size = sum(1 for _ in f if _.strip())

    # Rare-aware dropout vectors (optional)
    p_drop_vec = rare_mask_vec = None
    if args.rare_aware_dropout:
        if not (args.frag_vocab and args.frag_freq):
            raise ValueError("--rare-aware-dropout requires --frag-vocab and --frag-freq")
        p_drop_vec, rare_mask_vec = load_dropout_vectors_from_csv(
            args.frag_freq, args.frag_vocab,
            alpha=args.frag_drop_alpha, rare_quantile=args.frag_rare_quantile
        )

    # Dataset
    ds  = MolWithFragSetDataset(
        args.data, args.frags_jsonl, tok, args.max_len,
        max_frags_per_mol=args.max_frags_per_mol,
        max_frag_len=min(args.max_frag_len, args.max_len),
        frag_vocab_path=args.frag_vocab,
        frag_labels_pkl=args.frag_labels_pkl,
        frag_vocab_size=frag_vocab_size,
        legacy_fragment_order=args.legacy_fragment_order
    )

    n=len(ds); n_val=max(1,int(0.1*n)); n_train=n-n_val
    if n_train < 1 or args.batch_size < 1 or n_train < args.batch_size:
        raise ValueError(
            f'Training split has {n_train} examples and batch_size={args.batch_size}. '
            'Provide at least two molecules and choose a positive batch size '
            'no larger than the training split.'
        )
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate, drop_last=True,
        persistent_workers=(args.num_workers>0), prefetch_factor=(2 if args.num_workers>0 else None),
        worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
        persistent_workers=(args.num_workers>0), prefetch_factor=(2 if args.num_workers>0 else None),
        worker_init_fn=seed_worker
    )

    # Models
    vae = BaselineVAE(
        vocab_size=len(tok.id_to_tok), pad_id=tok.pad_id, bos_id=tok.bos_id, eos_id=tok.eos_id,
        emb=args.emb, hid=args.hid, z_dim=args.z, enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        dropout=args.dropout
    ).to(device)

    frag_enc = FragEncoder(
        vocab_size=len(tok.id_to_tok), pad_id=tok.pad_id,
        emb=args.emb, hid=args.hid, z_dim=args.z, layers=1, dropout=0.1
    ).to(device)

    # Optional BoW head
    bow_head=None; bce=None
    if args.lambda_frag_bow>0.0 and args.frag_labels_pkl and args.frag_vocab:
        assert frag_vocab_size is not None
        bow_head = nn.Linear(args.z, frag_vocab_size).to(device)
        pos_weight_vec = None
        if args.frag_freq:
            pos_weight_vec = load_pos_weight_from_csv(args.frag_freq, args.frag_vocab, clamp_max=args.bow_clamp_max).to(device)
            bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_vec)
        else:
            bce = nn.BCEWithLogitsLoss(reduction="none")
    elif args.lambda_frag_bow>0.0:
        args.log("WARNING: BoW enabled but missing --frag_labels_pkl/--frag_vocab; disabling BoW.")
        args.lambda_frag_bow = 0.0

    opt = torch.optim.AdamW(
        list(vae.parameters()) + list(frag_enc.parameters()) + ([] if bow_head is None else list(bow_head.parameters())),
        lr=args.lr, betas=(0.9,0.999), weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=USE_AMP)

    os.makedirs(args.outdir, exist_ok=True)
    global_step=0
    best_val_ce=float("inf")
    warned_no_mu=False

    if p_drop_vec is not None: p_drop_vec = p_drop_vec.to(device)
    if rare_mask_vec is not None: rare_mask_vec = rare_mask_vec.to(device)

    for epoch in range(1, args.epochs+1):
        vae.train(); frag_enc.train()
        pbar=tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        tr_ce_sum=0.0; tr_tok_sum=0; tr_kl_sum=0.0; tr_info_sum=0.0; tr_bow_sum=0.0
        n_info=0; n_bow=0; attn_H_sum=0.0; attn_H_n=0
        mean_valid_rows_sum=0.0; mean_valid_rows_n=0
        mean_H2p_sum=0.0; mean_H2p_n=0

        for batch in pbar:
            global_step += 1
            x_in, x_out, pad, frag_x, frag_pad, frag_idx, bows = (
                b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch
            )

            beta = min(1.0, float(global_step)/max(1, args.beta_steps))
            if args.gamma_info_warmup and args.gamma_info_warmup > 0:
                gamma_now = args.gamma_info * min(1.0, float(global_step)/float(args.gamma_info_warmup))
            else:
                gamma_now = args.gamma_info

            opt.zero_grad(set_to_none=True)
            with (autocast(**AMP_KW) if USE_AMP else contextlib.nullcontext()):
                ce, kl, logits, aux = vae(x_in, x_out, pad)

                kl_eff = torch.relu(kl - args.free_bits) if args.free_bits > 0.0 else kl
                mu = aux.get("mu", None) if isinstance(aux, dict) else None

                # fragment view + InfoNCE
                attn_w = None
                valid_rows = None
                if mu is not None:
                    out = frag_enc(
                        frag_x, frag_pad,
                        drop_frag_p=args.frag_dropout, training=True,
                        frag_idx=frag_idx, p_drop_vec=p_drop_vec,
                        keep_at_least_one_rare=True, rare_mask_vec=rare_mask_vec,
                        attn_query=mu, use_attn=args.attn_pooling,
                        attn_temp=args.attn_temp,
                        return_attn=True
                    )
                    if isinstance(out, tuple) and len(out) == 3:
                        z_frag, attn_w, valid_rows = out
                    elif isinstance(out, tuple) and len(out) == 2:
                        z_frag, attn_w = out
                    else:
                        z_frag = out

                    # Default: asymmetric InfoNCE (stop-grad on mu), unless --symmetric-info
                    mu_q = mu if args.symmetric_info else mu.detach()

                    if valid_rows is not None:
                        mask_info = (valid_rows >= 2)
                        info = info_nce(mu_q[mask_info], z_frag[mask_info], temperature=args.temperature) if mask_info.any() \
                               else torch.tensor(0.0, device=device)
                    else:
                        info = info_nce(mu_q, z_frag, temperature=args.temperature)
                else:
                    info = torch.tensor(0.0, device=device)
                    if not warned_no_mu:
                        args.log("WARNING: could not obtain mu from VAE; InfoNCE skipped for these steps.")
                        warned_no_mu=True

                loss = ce + beta*kl_eff + gamma_now*info

                # optional BoW on mu
                bow_loss = torch.tensor(0.0, device=device)
                if (args.lambda_frag_bow>0.0) and (bows is not None) and (bow_head is not None) and (mu is not None):
                    logits_bow = bow_head(mu)           # (B, V)
                    raw = bce(logits_bow, bows)         # (B, V)
                    bow_loss = raw.mean()
                    loss = loss + args.lambda_frag_bow * bow_loss

                # attention entropy (for logging; apply reg only if coeff>0)
                if args.attn_pooling and attn_w is not None:
                    p = attn_w.float().clamp_min(1e-12)
                    H_all = (-p * p.log()).sum(dim=1)  # (B,)
                    H = H_all.mean()
                    if args.attn_entropy > 0.0:
                        loss = loss - args.attn_entropy * H
                else:
                    H = None; H_all = None

            # stats
            with torch.no_grad():
                tok_count=(~pad).sum().item()
                tr_ce_sum += ce.item()*tok_count
                tr_tok_sum += tok_count
                tr_kl_sum += kl.item()*x_in.size(0)
                tr_info_sum += info.item(); n_info += 1
                tr_bow_sum += bow_loss.item(); n_bow += 1
                if H is not None:
                    attn_H_sum += H.item(); attn_H_n += 1
                if valid_rows is not None:
                    mean_valid_rows_sum += valid_rows.float().mean().item(); mean_valid_rows_n += 1
                    if H_all is not None:
                        mask_2p = (valid_rows >= 2)
                        if mask_2p.any():
                            mean_H2p_sum += H_all[mask_2p].mean().item()
                            mean_H2p_n += 1

            # step
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    list(vae.parameters())+list(frag_enc.parameters())+([] if bow_head is None else list(bow_head.parameters())),
                    1.0
                )
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(vae.parameters())+list(frag_enc.parameters())+([] if bow_head is None else list(bow_head.parameters())),
                    1.0
                )
                opt.step()

            if args.log_interval>0 and global_step % args.log_interval == 0:
                ppl = safe_exp(ce.item())
                pbar.set_postfix(ce=f"{ce.item():.3f}", kl=f"{kl.item():.3f}",
                                 info=f"{info.item():.3f}", bow=f"{bow_loss.item():.3f}",
                                 beta=f"{beta:.2f}", gamma=f"{gamma_now:.4f}", ppl=f"{ppl:.2f}")

            if args.save_interval>0 and global_step % args.save_interval == 0:
                ckpt_path = os.path.join(args.outdir, f"ckpt_step{global_step}.pt")
                save_checkpoint(ckpt_path, vae, opt, global_step, args)

        # ---- Validation (CE/KL only; CE averaged per-token) ----
        vae.eval(); frag_enc.eval()
        val_ce_sum, val_tok_sum, val_kl_sum = 0.0, 0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_in, x_out, pad, frag_x, frag_pad, frag_idx, _ = (
                    b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch
                )
                ce, kl, _, _ = vae(x_in, x_out, pad)
                tok_count=(~pad).sum().item()
                val_ce_sum += ce.item()*tok_count
                val_tok_sum += tok_count
                val_kl_sum  += kl.item()*x_in.size(0)

        train_ce = tr_ce_sum / max(1, tr_tok_sum)
        train_ppl= safe_exp(train_ce)
        train_kl = tr_kl_sum / max(1, len(train_loader.dataset))
        train_info = tr_info_sum / max(1, n_info)
        train_bow  = tr_bow_sum / max(1, n_bow)
        mean_attn_H = (attn_H_sum / max(1, attn_H_n)) if attn_H_n>0 else float('nan')
        mean_valid_rows = (mean_valid_rows_sum / max(1, mean_valid_rows_n)) if mean_valid_rows_n>0 else float('nan')
        mean_H2p = (mean_H2p_sum / max(1, mean_H2p_n)) if mean_H2p_n>0 else float('nan')

        val_ce = val_ce_sum / max(1, val_tok_sum)
        val_ppl= safe_exp(val_ce)
        val_kl = val_kl_sum / max(1, len(val_loader.dataset))

        args.log(f"[VAL] epoch {epoch}: "
                 f"trainCE {train_ce:.3f} | trainPPL {train_ppl:.2f} | trainKL {train_kl:.3f} | "
                 f"trainInfo {train_info:.3f} | trainBoW {train_bow:.3f} | "
                 f"attnH {mean_attn_H:.3f} | meanF {mean_valid_rows:.2f} | H2+ {mean_H2p:.3f} || "
                 f"valCE {val_ce:.3f} | valPPL {val_ppl:.2f} | valKL {val_kl:.3f}")

        # MLflow metrics
        try:
            if ml is not None:
                ml.log_metrics({
                    "train/ce": train_ce, "train/ppl": train_ppl, "train/kl": train_kl,
                    "train/info": train_info, "train/bow": train_bow,
                    "train/attn_entropy": mean_attn_H,
                    "train/mean_valid_frags": mean_valid_rows,
                    "train/attn_entropy_ge2": mean_H2p,
                    "val/ce": val_ce, "val/ppl": val_ppl, "val/kl": val_kl,
                    "sched/gamma_now": (args.gamma_info if not args.gamma_info_warmup else
                                        args.gamma_info * min(1.0, float(global_step)/float(args.gamma_info_warmup)))
                }, step=epoch)
        except Exception:
            pass

        if val_ce < best_val_ce:
            best_val_ce = val_ce
            ckpt_path = os.path.join(args.outdir, "best.pt")
            save_checkpoint(ckpt_path, vae, opt, global_step, args)
            args.log(f"Saved best checkpoint → {ckpt_path}")
            try:
                if ml is not None:
                    ml.log_artifact(ckpt_path, artifact_path="checkpoints")
            except Exception as e:
                args.log(f"[MLflow] Failed to log artifact: {e}")

    args.log("Training complete.")
    try:
        if ml is not None:
            ml.end_run()
    except Exception:
        pass

if __name__ == "__main__":
    main()
