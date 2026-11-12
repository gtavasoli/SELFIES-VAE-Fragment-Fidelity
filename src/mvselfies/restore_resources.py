"""Extract resource bundles using the repository's current folder names."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


def restore_bundle(archive, root):
    """Extract one dataset, model, or result archive into a repository root.

    Parameters
    ----------
    archive : pathlib.Path
        ``datasets.zip``, ``models.zip``, or ``results.zip`` downloaded from
        Zenodo. Archive checksums should be verified before extraction.
    root : pathlib.Path
        Repository root. Existing documentation, manifests, inventory, and summary
        are preserved during extraction.

    Notes
    -----
    Earlier archive folder names ``paper`` and ``paper_tables`` map to
    ``main`` and ``tables``. Resource file contents are copied unchanged.
    Both archived and current folder names are accepted. Each resource is
    replaced atomically after its archive entry has been read successfully.

    Raises
    ------
    ValueError
        If the bundle name is unknown or a file path leaves its bundle folder.
    """
    bundle = archive.stem
    if bundle not in {'datasets', 'models', 'results'}:
        raise ValueError(f'Unknown resource bundle: {archive.name}')
    bundle_root = (root / bundle).resolve()
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            if item.is_dir():
                continue
            parts = list(PurePosixPath(item.filename).parts)
            if (len(parts) < 2 or parts[0] != bundle
                    or any(part in {'.', '..'} or ':' in part or '\\' in part
                           for part in parts)):
                raise ValueError(f'Invalid resource path: {item.filename}')
            if len(parts) == 2 and parts[1] in {
                    'README.md', 'manifest.json', 'inventory.json', 'summary.csv'}:
                continue
            if bundle in {'models', 'results'}:
                parts[1] = {'paper': 'main', 'paper_tables': 'tables'}.get(parts[1], parts[1])
            destination = root.joinpath(*parts).resolve()
            if not destination.is_relative_to(bundle_root):
                raise ValueError(f'Resource path leaves bundle folder: {item.filename}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as outgoing:
                    temporary = Path(outgoing.name)
                    with source.open(item) as incoming:
                        shutil.copyfileobj(incoming, outgoing)
                temporary.replace(destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def main():
    """Restore the specified resource ZIPs into the current working directory."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    args = parser.parse_args()
    for archive in args.archives:
        restore_bundle(archive, Path.cwd())
        print(f'Restored {archive.name}')


if __name__ == '__main__':
    main()
