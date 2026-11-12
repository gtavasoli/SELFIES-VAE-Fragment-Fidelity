"""Verify downloaded resource checksums, checkpoint loading, and archived metrics."""
import argparse
import hashlib
import json
from pathlib import Path


def verify_manifest(root, name):
    """Check every resource file against a bundle's byte counts and SHA-256 values.

    Parameters
    ----------
    root : pathlib.Path
        Repository root containing the extracted resource folders.
    name : str
        Bundle folder name, normally ``datasets``, ``models``, or ``results``.
        Manifest entries are relative to ``root``, not the bundle directory.

    Raises
    ------
    ValueError
        If a resolved path leaves the root or a size/checksum differs.
    FileNotFoundError
        If the manifest or one of its listed files is absent.
    """
    manifest = json.loads((root/name/'manifest.json').read_text(encoding='utf-8'))
    for item in manifest:
        path = (root/item['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Manifest path leaves resource root: {path}')
        if path.stat().st_size != item['bytes']:
            raise ValueError(f'File size mismatch: {path}')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != item['sha256']:
            raise ValueError(f'SHA-256 mismatch: {path}')
    print(f'{name}: verified {len(manifest)} files')


def main():
    """Verify all bundles, load catalog checkpoints on CPU, and check saved metrics."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    for name in ['datasets','models','results']:
        verify_manifest(args.root, name)
    import torch
    from .generate import load_model
    from .summarize import summarize
    runs = json.loads((args.root/'experiments/runs.json').read_text(encoding='utf-8'))
    for run in runs:
        load_model(args.root/run['checkpoint'], torch.device('cpu'))
    summarize(args.root, runs, verify=True)
    print(f'Loaded all {len(runs)} checkpoints and verified all archived fragment metrics.')


if __name__ == '__main__':
    main()
