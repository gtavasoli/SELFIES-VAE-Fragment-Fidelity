"""Train Baseline-VAE with reconstruction loss and a fixed unit KL coefficient.

The CLI makes a seed-controlled 90/10 split, selects ``best.pt`` by validation
cross-entropy, and optionally records training metrics with MLflow.
"""
import os, math, random, datetime, contextlib, argparse, json
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

# ---------------- Logger shim ----------------
def _make_logger(log_path=None):
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

# ---------------- Tokenizer ----------------
# ---------------- Dataset ----------------
class SelfiesFileDataset(Dataset):
    """Load nonempty molecular SELFIES rows and encode them on demand.

    Parameters
    ----------
    selfies_path : str
        UTF-8 molecular SELFIES file, one molecule per nonempty line.
    tokenizer : SelfiesTokenizer
        Ordered vocabulary shared with the molecular VAE.
    max_len : int
        Fixed length for shifted token sequences and their padding masks.
    """
    def __init__(self, selfies_path: str, tokenizer: SelfiesTokenizer, max_len: int):
        with open(selfies_path, "r", encoding="utf-8") as f:
            self.data = [line.strip() for line in f if line.strip()]
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        """Return long input/target tensors and a boolean target-padding mask."""
        sel = self.data[idx]
        x_in, x_out, pad_mask = self.tok.encode(sel, self.max_len)
        return (torch.tensor(x_in, dtype=torch.long),
                torch.tensor(x_out, dtype=torch.long),
                torch.tensor(pad_mask, dtype=torch.bool))

def collate(batch):
    """Stack dataset triples into input, target, and mask tensors of shape (B, T)."""
    x_in  = torch.stack([b[0] for b in batch], 0)
    x_out = torch.stack([b[1] for b in batch], 0)
    pad   = torch.stack([b[2] for b in batch], 0)
    return x_in, x_out, pad

# ---------------- Utils ----------------
def set_seed(seed:int):
    """Seed Python and PyTorch CPU/CUDA RNGs and set PYTHONHASHSEED for children."""
    random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def decode_samples(model, tok, device, out_path: str, n: int, max_len: int):
    """Write ``n`` prior samples as SELFIES and restore the model's training mode.

    Sampling uses temperature 1.0, top-p 0.9, and repetition penalty 1.1.
    The output is raw decoded SELFIES, without molecular validation or deduplication.
    """
    was_training = model.training
    try:
        model.eval()
        z = torch.randn(n, model.z_dim, device=device)
        ids = model.sample(z, max_len=max_len, temperature=1.0, top_p=0.9, repetition_penalty=1.1)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ids.tolist():
                f.write(tok.decode_ids(row) + "\n")
    finally:
        if was_training:
            model.train()

def save_checkpoint(path, model, optimizer, step, args):
    """Save VAE/optimizer state, step, ordered vocabulary, and primitive CLI values.

    The checkpoint's embedded vocabulary preserves token indices for inference.
    Nonprimitive argument values, such as logger callables, are omitted.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    safe_args = {}
    for k, v in vars(args).items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe_args[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str,int,float,bool,type(None))) for x in v):
            safe_args[k] = v
    torch.save({"model_state": model.state_dict(),
                "vocab": SelfiesTokenizer(args.vocab).id_to_tok,
                "opt_state": optimizer.state_dict(),
                "step": step,
                "args": safe_args}, path)

# ---------------- MLflow helpers ----------------
def _maybe_import_mlflow(enabled: bool):
    """Return the optional tracking module, or None when disabled/unavailable."""
    if not enabled:
        return None
    try:
        import mlflow
        return mlflow
    except Exception:
        print("[WARN] MLflow requested but not installed; continuing without tracking.")
        return None

def _safe_log_params(mlflow, params: dict):
    """Serialize container/object parameter values before sending them to MLflow."""
    # flatten basic params
    flat = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            flat[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str,int,float,bool,type(None))) for x in v):
            flat[k] = json.dumps(v)
        else:
            flat[k] = str(v)
    mlflow.log_params(flat)

# ---------------- CLI ----------------
def parse_args():
    """Return the parsed training namespace and its configured logger callable."""
    p = argparse.ArgumentParser("Train Baseline-VAE (optional MLflow tracking)")
    # data / model
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--vocab", type=str, required=True)
    p.add_argument("--max-len", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--emb", type=int, default=256)
    p.add_argument("--hid", type=int, default=512)
    p.add_argument("--z", type=int, default=64)
    p.add_argument("--enc-layers", type=int, default=2)
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--sample-interval", type=int, default=2000)
    p.add_argument("--save-interval", type=int, default=2000)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-file", type=str, default=None)
    p.add_argument("--sample-n", type=int, default=16, help="num samples to decode when logging")

    # mlflow
    p.add_argument("--mlflow", action="store_true", help="enable MLflow tracking")
    p.add_argument("--mlflow-uri", type=str, default=None, help="tracking URI (e.g., http://localhost:5000)")
    p.add_argument("--mlflow-experiment", type=str, default="Baseline-VAE", help="experiment name")
    p.add_argument("--run-name", type=str, default=None, help="custom MLflow run name")
    p.add_argument("--mlflow-tags", type=str, default=None, help='comma-separated key=val,...')

    args = p.parse_args()
    log = _make_logger(args.log_file)
    if not callable(log):
        log = lambda *m, **k: print(*m)
    return args, log

# ---------------- Train ----------------
def main():
    """Train the baseline, validate each epoch, and save checkpoints/sample files."""
    args, log = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Tokenizer & data
    tokenizer = SelfiesTokenizer(args.vocab)
    ds  = SelfiesFileDataset(args.data, tokenizer, args.max_len)

    # 90/10 split
    n = len(ds); n_val = max(1, int(0.1*n)); n_train = n - n_val
    if n_train < 1 or args.batch_size < 1 or n_train < args.batch_size:
        raise ValueError(
            f'Training split has {n_train} examples and batch_size={args.batch_size}. '
            'Provide at least two molecules and choose a positive batch size '
            'no larger than the training split.'
        )
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val],
                                                     generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate, drop_last=True,
        persistent_workers=(args.num_workers > 0), prefetch_factor=(2 if args.num_workers>0 else None)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
        persistent_workers=(args.num_workers > 0), prefetch_factor=(2 if args.num_workers>0 else None)
    )

    # Model/opt
    model = BaselineVAE(
        vocab_size=len(tokenizer.id_to_tok), pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
        emb=args.emb, hid=args.hid, z_dim=args.z, enc_layers=args.enc_layers,
        dec_layers=args.dec_layers, dropout=args.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=USE_AMP)

    os.makedirs(args.outdir, exist_ok=True)
    global_step, best_val_ce = 0, float("inf")

    # --- MLflow setup ---
    mlflow = _maybe_import_mlflow(args.mlflow)
    run_ctx = contextlib.nullcontext()
    if mlflow:
        if args.mlflow_uri: mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        tags = {}
        if args.mlflow_tags:
            for kv in args.mlflow_tags.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    tags[k.strip()] = v.strip()
        run_ctx = mlflow.start_run(run_name=args.run_name, tags=tags)
    # ---------------------

    with run_ctx:
        # Log params once
        if mlflow:
            _safe_log_params(mlflow, {
                "data": args.data, "vocab": args.vocab, "max_len": args.max_len,
                "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr,
                "weight_decay": args.weight_decay, "emb": args.emb, "hid": args.hid, "z": args.z,
                "enc_layers": args.enc_layers, "dec_layers": args.dec_layers, "dropout": args.dropout,
                "seed": args.seed, "num_workers": args.num_workers, "sample_interval": args.sample_interval,
                "save_interval": args.save_interval, "log_interval": args.log_interval
            })
            # Useful extra params
            mlflow.log_param("vocab_size", len(tokenizer.id_to_tok))
            mlflow.log_param("device", str(device))

        for epoch in range(1, args.epochs+1):
            model.train()
            running_loss = 0.0
            running_ce   = 0.0
            running_kl   = 0.0
            token_count  = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
            for batch in pbar:
                global_step += 1
                x_in, x_out, pad_mask = (b.to(device, non_blocking=True) for b in batch)

                opt.zero_grad(set_to_none=True)
                with (autocast(**AMP_KW) if USE_AMP else contextlib.nullcontext()):
                    ce, kl, logits, _ = model(x_in, x_out, pad_mask)
                    loss = ce + kl * 1.0

                if USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                # accumulators for epoch averages (token-weighted CE)
                tok_count = (~pad_mask).sum().item()
                running_loss += loss.item() * tok_count
                running_ce   += ce.item()   * tok_count
                running_kl   += kl.item()   * x_in.size(0)
                token_count  += tok_count

                if args.log_interval > 0 and global_step % args.log_interval == 0:
                    ppl = math.exp(max(1e-8, ce.item()))
                    pbar.set_postfix(ce=f"{ce.item():.3f}", kl=f"{kl.item():.3f}", ppl=f"{ppl:.2f}")
                    if mlflow:
                        mlflow.log_metrics({"train/ce_batch": ce.item(),
                                            "train/kl_batch": kl.item(),
                                            "train/ppl_batch": ppl,
                                            "train/loss_batch": loss.item()},
                                           step=global_step)

                if args.sample_interval > 0 and global_step % args.sample_interval == 0:
                    out_path = os.path.join(args.outdir, f"samples_step{global_step}.selfies")
                    decode_samples(model, tok=tokenizer, device=device, out_path=out_path,
                                   n=args.sample_n, max_len=args.max_len)
                    if mlflow and os.path.exists(out_path):
                        mlflow.log_artifact(out_path, artifact_path="samples")

                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    ckpt_path = os.path.join(args.outdir, f"ckpt_step{global_step}.pt")
                    save_checkpoint(ckpt_path, model, opt, global_step, args)
                    if mlflow and os.path.exists(ckpt_path):
                        mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

            # --- Validation (epoch) ---
            model.eval()
            val_ce_sum, val_tok_sum, val_kl_sum = 0.0, 0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_in, x_out, pad_mask = (b.to(device, non_blocking=True) for b in batch)
                    ce, kl, _, _ = model(x_in, x_out, pad_mask)
                    tok_count = (~pad_mask).sum().item()
                    val_ce_sum += ce.item()*tok_count
                    val_tok_sum += tok_count
                    val_kl_sum += kl.item()*x_in.size(0)

            # epoch-level metrics
            train_ce = running_ce / max(1, token_count)
            train_ppl = math.exp(max(1e-8, train_ce))
            train_kl = running_kl / max(1, len(train_loader.dataset))
            train_loss = running_loss / max(1, token_count)

            val_ce = val_ce_sum / max(1, val_tok_sum)
            val_ppl = math.exp(max(1e-8, val_ce))
            val_kl = val_kl_sum / max(1, len(val_loader.dataset))

            # console + MLflow
            msg = f"[VAL] epoch {epoch}: trainCE {train_ce:.3f} | trainPPL {train_ppl:.2f} | trainKL {train_kl:.3f} || valCE {val_ce:.3f} | valPPL {val_ppl:.2f} | valKL {val_kl:.3f}"
            log(msg)
            if mlflow:
                mlflow.log_metrics({
                    "train/ce": train_ce, "train/ppl": train_ppl, "train/kl": train_kl, "train/loss": train_loss,
                    "val/ce": val_ce, "val/ppl": val_ppl, "val/kl": val_kl
                }, step=global_step)

            # best checkpoint
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                ckpt_path = os.path.join(args.outdir, "best.pt")
                save_checkpoint(ckpt_path, model, opt, global_step, args)
                print(f"Saved best checkpoint → {ckpt_path}")
                if mlflow and os.path.exists(ckpt_path):
                    mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

        # final sample dump
        final_samples = os.path.join(args.outdir, "samples_final.selfies")
        decode_samples(model, tokenizer, device, final_samples, n=max(args.sample_n, 32), max_len=args.max_len)
        if mlflow and os.path.exists(final_samples):
            mlflow.log_artifact(final_samples, artifact_path="samples")

        # also store the vocab used (handy for reproducibility)
        if mlflow and os.path.exists(args.vocab):
            mlflow.log_artifact(args.vocab, artifact_path="data")

if __name__ == "__main__":
    main()
