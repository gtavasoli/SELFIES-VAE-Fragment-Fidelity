"""Sample a released VAE checkpoint and write canonical SMILES and metrics."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import selfies as sf
import torch
from rdkit import Chem
from tqdm import tqdm

from .model import BaselineVAE
from .tokenizer import SelfiesTokenizer


def canonical_smiles(smiles):
    """Return canonical isomeric SMILES, or None for empty/unparseable input."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None


def load_model(checkpoint_path, device, vocab_path=None):
    """Load VAE weights and their ordered vocabulary for inference.

    Parameters
    ----------
    checkpoint_path : str or path-like
        Checkpoint containing ``model_state``, training ``args``, and optionally
        an embedded ``vocab`` list.
    device : torch.device or str
        Device used for loading weights and running the returned model.
    vocab_path : str or path-like, optional
        External vocabulary, required if the checkpoint has none. If both are
        present, their entries must match in order.

    Returns
    -------
    model : BaselineVAE
        Strictly loaded molecular backbone in evaluation mode.
    tok : SelfiesTokenizer
        Tokenizer matching the checkpoint's embedding indices.
    args : dict
        Saved architecture and training settings.

    Raises
    ------
    ValueError
        If a vocabulary is missing or conflicts with the embedded vocabulary.
    RuntimeError
        If weights do not match the reconstructed architecture.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = checkpoint.get('args', {})
    if vocab_path:
        tok = SelfiesTokenizer(vocab_path)
        if 'vocab' in checkpoint and tok.id_to_tok != checkpoint['vocab']:
            raise ValueError('The supplied vocabulary differs from the checkpoint vocabulary.')
    elif 'vocab' in checkpoint:
        tok = SelfiesTokenizer.from_tokens(checkpoint['vocab'])
    else:
        raise ValueError('This older checkpoint requires --vocab.')
    model = BaselineVAE(
        vocab_size=len(tok.id_to_tok), pad_id=tok.pad_id,
        bos_id=tok.bos_id, eos_id=tok.eos_id,
        emb=args.get('emb', 256), hid=args.get('hid', 512),
        z_dim=args.get('z', 64), enc_layers=args.get('enc_layers', 2),
        dec_layers=args.get('dec_layers', 3), dropout=args.get('dropout', 0.2),
    ).to(device)
    model.load_state_dict(checkpoint.get('model_state', checkpoint), strict=True)
    model.eval()
    return model, tok, args


def training_smiles(path, is_selfies=False):
    """Read a unique canonical training reference for novelty calculations.

    ``path=None`` returns an empty set. With ``is_selfies=True``, decode each
    line before canonicalization. Invalid lines are skipped; a supplied file
    with no valid molecules raises ValueError.
    """
    canonical = set()
    if path is None:
        return canonical
    with open(path, encoding='utf-8') as stream:
        for line in tqdm(stream, desc='Training reference'):
            try:
                text = line.strip()
                smiles = sf.decoder(text) if is_selfies else text
                can = canonical_smiles(smiles)
                if can:
                    canonical.add(can)
            except (sf.DecoderError, ValueError):
                continue
    if not canonical:
        raise ValueError('The supplied training reference contains no valid molecules.')
    return canonical


@torch.inference_mode()
def generate(model, tok, n, max_len, batch_size=128, temperature=1.0, top_p=0.9,
             repetition_penalty=1.1):
    """Yield canonical SMILES or None for each prior-sampling attempt.

    Parameters
    ----------
    model : BaselineVAE
        Model already placed on its inference device and in evaluation mode.
    tok : SelfiesTokenizer
        Vocabulary used to decode sampled token indices.
    n : int
        Number of attempts, including invalid and duplicate molecules.
    max_len, batch_size : int
        Token-length limit and maximum number of simultaneous latent samples.
    temperature, top_p, repetition_penalty : float
        Sampling settings forwarded to ``model.sample``.

    Yields
    ------
    str or None
        Canonical isomeric SMILES, or None when SELFIES decoding or molecular
        validation fails. No deduplication is performed here.
    """
    device = next(model.parameters()).device
    for start in tqdm(range(0, n, batch_size), desc='Sampling'):
        z = torch.randn(min(batch_size, n-start), model.z_dim, device=device)
        ids = model.sample(z, max_len, temperature, top_p, repetition_penalty)
        for row in ids.cpu().tolist():
            try:
                yield canonical_smiles(sf.decoder(tok.decode_ids(row)))
            except (sf.DecoderError, ValueError):
                yield None


def main():
    """Sample the CLI checkpoint and write valid SMILES plus a metrics JSON file."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--vocab', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--max-len', type=int)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    parser.add_argument('--seed', type=int, default=1)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument('--train-smiles', type=Path)
    reference.add_argument('--train-selfies', type=Path)
    args = parser.parse_args()
    if args.n < 1 or args.batch_size < 1 or (args.max_len is not None and args.max_len < 2):
        parser.error('n and batch-size must be positive; max-len must be at least 2.')
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error('temperature must be positive and top-p must be in (0, 1].')
    inputs = [args.ckpt, args.vocab, args.train_smiles, args.train_selfies]
    for output in [args.out, args.out.with_suffix('.metrics.json')]:
        for source in inputs:
            if source is not None and (
                output.resolve() == source.resolve()
                or (output.exists() and source.exists() and output.samefile(source))
            ):
                parser.error('Output paths must differ from checkpoint, vocabulary, and training inputs.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tok, config = load_model(args.ckpt, device, args.vocab)
    max_len = args.max_len or config.get('max_len', 84)
    train = training_smiles(args.train_smiles or args.train_selfies, bool(args.train_selfies))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    unique = set()
    n_valid = 0
    with args.out.open('w', encoding='utf-8') as stream:
        for smiles in generate(model, tok, args.n, max_len, args.batch_size,
                               args.temperature, args.top_p, args.repetition_penalty):
            if smiles:
                stream.write(smiles + '\n')
                unique.add(smiles)
                n_valid += 1
    metrics = {
        'n_total': args.n, 'n_valid': n_valid, 'n_unique': len(unique),
        'validity': n_valid/args.n, 'uniqueness': len(unique)/max(1, n_valid),
        'novelty': len(unique-train)/max(1, len(unique)) if train else None,
        'seed': args.seed, 'batch_size': args.batch_size, 'max_len': max_len,
        'temperature': args.temperature, 'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
    }
    args.out.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
