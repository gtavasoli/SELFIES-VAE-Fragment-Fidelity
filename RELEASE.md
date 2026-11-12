# Resource release 0.1.0

Version 0.1.0 is publicly available on
[Zenodo](https://doi.org/10.5281/zenodo.24147562).

| File | Contents |
| --- | --- |
| [datasets.zip](https://zenodo.org/records/24147562/files/datasets.zip?download=1) | Processed datasets, vocabularies, fragments, labels, inventory, and manifest |
| [models.zip](https://zenodo.org/records/24147562/files/models.zip?download=1) | All 18 generative checkpoints and their manifest |
| [results.zip](https://zenodo.org/records/24147562/files/results.zip?download=1) | Generated molecules, evaluations, count tables, logs, summaries, and manifest |
| [SHA256SUMS.txt](https://zenodo.org/records/24147562/files/SHA256SUMS.txt?download=1) | Checksums for the three ZIP archives |

Archive sizes are listed in the [README](README.md#resources-and-provenance).
A copy of [SHA256SUMS.txt](SHA256SUMS.txt) is included with the code.
Each ZIP preserves its top-level folder and includes the resource documentation.
Large resource files and downloaded archives remain ignored by Git.

## Download and verify

Follow the [installation instructions](README.md#install) using Python 3.11.
Download the three archives to the repository root, along with `SHA256SUMS.txt`.
On Windows PowerShell, display the archive hashes and compare them with that file:

```powershell
Get-FileHash datasets.zip, models.zip, results.zip -Algorithm SHA256
```

On Linux, check them directly:

```bash
sha256sum -c SHA256SUMS.txt
```

Then extract and check the resources:

```console
python src/mvselfies/restore_resources.py datasets.zip models.zip results.zip

python -m mvselfies.verify

python -m mvselfies.generate --ckpt models/main/Rare-Aware-VAE_1m_s1/best.pt --out runs/reviewer/generated.smi --n 100

python -m mvselfies.summarize --verify --out runs/reviewer/summary.csv
```

The archives use `models/main/`, `results/main/`, and `results/tables/`,
matching the experiment catalog. The extractor preserves repository documentation,
manifests, inventory, and summary files. Pass a single ZIP filename to restore
one bundle.

Verification checks file hashes, loads every VAE checkpoint, and recomputes
fragment metrics from saved count tables. Recorded settings and interpretation
notes are documented in the [model](models/README.md#recorded-training-settings),
[dataset](datasets/README.md), and [result](results/README.md) bundle READMEs.

## Licenses and future versions

Model weights, software, and original experimental results use the
[MIT License](LICENSE). ChEMBL-derived datasets and training-reference data
retain CC BY-SA 3.0, with attribution as described in the
[dataset README](datasets/README.md).
