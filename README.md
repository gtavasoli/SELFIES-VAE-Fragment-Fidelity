# SELFIES-VAE Fragment Fidelity

Code and archived resources accompanying **Fragment-Distribution Fidelity in SELFIES VAEs: Contrastive Bias and Rare-Aware Regularization Across Scales**.

The package implements three model configurations:

| Model | Training objective |
| --- | --- |
| Baseline-VAE | SELFIES reconstruction and KL |
| Contrastive-VAE | VAE plus molecule-fragment InfoNCE |
| Rare-Aware-VAE | Fragment dropout, attention pooling, optional fragment BoW |

All three configurations share the `BaselineVAE` backbone. Training arguments and artifact provenance are recorded in the [experiment catalog](experiments/runs.json). The released comparison includes Contrastive-VAE's negative results.

## Resources and provenance

| Folder | Contents | Stored in Git |
| --- | --- | --- |
| `src/mvselfies/` | Python experiment package | Yes |
| `experiments/runs.json` | Exact archived training arguments and artifact provenance | Yes |
| `tests/` | Tests using small synthetic datasets | Yes |
| `datasets/` | Processed molecule sequences, vocabularies, fragment mappings and labels | Documentation and inventories |
| `models/` | Generative checkpoints with embedded SELFIES vocabularies | Documentation and checksums |
| `results/` | Generated molecules, properties, fragment counts, reports and logs | Documentation, checksums and summary tables |
| `runs/` | Outputs from new experiments | No |

Resource version **0.1.0** is publicly available on [Zenodo](https://doi.org/10.5281/zenodo.24147562).

| Bundle | Compressed size | Download |
| --- | --- | --- |
| Datasets | 89.8 MB | [datasets.zip](https://zenodo.org/records/24147562/files/datasets.zip?download=1) |
| Trained models | 586.1 MB | [models.zip](https://zenodo.org/records/24147562/files/models.zip?download=1) |
| Results | 30.5 MB | [results.zip](https://zenodo.org/records/24147562/files/results.zip?download=1) |

Archive SHA-256 checksums are in [SHA256SUMS.txt](SHA256SUMS.txt), also available [on Zenodo](https://zenodo.org/records/24147562/files/SHA256SUMS.txt?download=1).

Download the archives to the repository root, verify their checksums, then extract them there:

```console
python src/mvselfies/restore_resources.py datasets.zip models.zip results.zip
```

The extractor writes the resource files to `datasets/`, `models/`, and `results/` and preserves repository documentation and manifests. It requires only the Python standard library. See [resource release instructions](RELEASE.md) for checksum verification and archive layout details.

`models/main/` and `results/main/` contain nine runs matching the manuscript's main table. `models/replicates/` and `results/replicates/` contain nine later 1M-scale runs, three per model. These groups have different training budgets, and both include a run labelled seed 1. They must not be treated as four interchangeable independent seeds. The full inventory is in [results/summary.csv](results/summary.csv).

## Install

Use Python **3.11**. From the repository root:

```console
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
```

```bash
# Linux / macOS
source .venv/bin/activate
export PYTHONUTF8=1
```

```console
python -m pip install -r requirements.txt
```

`requirements.txt` installs the local package in editable mode. Dependencies and version constraints are declared in [pyproject.toml](pyproject.toml).

Core versions are pinned to the main recorded environment: PyTorch 2.8.0, RDKit 2025.3.6, SELFIES 2.1.1, NumPy 1.26.4, and tqdm 4.67.1. The optional MLflow integration can be installed with `python -m pip install -e ".[tracking]"`. CUDA training needs a PyTorch build appropriate to the machine; recorded CUDA and library versions are in each run's `env.txt`.

The later 1M runs recorded PyTorch 2.9.0 and RDKit 2025.9.1. Numerical equivalence under the pinned main-run environment has not been tested.

## Verify the downloaded resources

Extract all three resource bundles at the repository root before running the full verification:

```console
python -m mvselfies.verify
```

Verification checks resource sizes and SHA-256 hashes, loads each released checkpoint on CPU with strict state matching, and recomputes fragment metrics from the saved count tables. These checks assess file integrity, checkpoint compatibility, and metric consistency.

## Generate molecules without training

Only the model bundle is required for this example. Each checkpoint contains its own vocabulary and architecture settings:

```console
python -m mvselfies.generate --ckpt models/main/Rare-Aware-VAE_1m_s1/best.pt --out runs/demo/generated.smi --n 100 --batch-size 32 --seed 1
```

The output includes canonical valid SMILES and `generated.metrics.json` with validity and uniqueness. Add `--train-smiles datasets/1m/clean_smiles.txt` to calculate novelty. Sampling is batched to limit GPU memory. The original generated molecules are included in the results bundle; a new sampled file is not expected to be byte-identical to the archived file.

Released multiview checkpoints support inference but omit the auxiliary fragment
encoder and BoW head needed for complete training resumption. See the
[model bundle documentation](models/README.md) for checkpoint contents.

## Run recorded experiments

The `main` group contains the nine main-table experiments; `replicates` contains the nine additional 1M runs. List the configurations and preview a training command:

```console
python -m mvselfies.experiments --list
python -m mvselfies.experiments --run main/Baseline-VAE_10k_s1 --stage train --dry-run
```

To retrain and then generate, analyze, and evaluate:

```console
python -m mvselfies.experiments --run main/Baseline-VAE_10k_s1 --stage all
```

The runner reads training settings from the catalog. The main 10k runs use 149 epochs; the main 100k and 1M runs use 15. Outputs for this example go under `runs/main/Baseline-VAE_10k_s1/`. If a trained checkpoint already exists there, select another output directory with `--runs-dir`.

Later 1M runs use 15 epochs for seed 1 and two epochs for seeds 10 and 100.

Stages can be run individually with `--stage train`, `generate`, `analyze`, or `evaluate`. The latter stages use a checkpoint and outputs under `runs/`. For inference directly from a released checkpoint, use the generation command above.

For archived Contrastive-VAE/Rare-Aware-VAE runs, the runner explicitly enables historical positional fragment pairing. This reproduces the archived input arrangement and emits a warning. Direct use of `mvselfies.train_multiview` rejects mismatched molecule/fragment identities unless `--legacy-fragment-order` is supplied. Use freshly aligned fragments for new experiments, and report them separately from historical results.

Rare-Aware-VAE uses a fragment BoW head at 10k/100k; it is disabled at 1M. BoW labels are expanded per item to avoid a corpus-sized dense matrix.

Recorded gradient flow, dropout behavior, and split settings are described in the
[model bundle documentation](models/README.md#recorded-training-settings).

## Reproduce tables from saved counts

```console
python -m mvselfies.summarize --verify --out runs/verified_summary.csv
```

Main-table source CSVs are provided under `results/tables/`. JSD is the square root of Jensen-Shannon divergence using natural logarithms. Rare recall uses positive-count fragments at or below the tenth-percentile training-count threshold, including ties; the selected set can exceed 10% of fragment types. Top-k mass is calculated within each distribution's own ranked fragments. Additional evaluation details are in the [result bundle documentation](results/README.md).

## Prepare a new dataset

Use a new directory so the archived data remain intact:

```console
python -m mvselfies.sanity --input-txt my_smiles.txt --outdir runs/new_data/sanity --stereo --salt-split

python -m mvselfies.extract_fragments --train-smiles runs/new_data/sanity/clean_smiles.txt --out-jsonl runs/new_data/fragments.jsonl --out-vocab runs/new_data/frag_vocab.txt --out-freq runs/new_data/frag_freq.csv --max-workers 2

python -m mvselfies.fill_fragments --frags-jsonl runs/new_data/fragments.jsonl --frag-vocab runs/new_data/frag_vocab.txt --out-jsonl runs/new_data/fragments_filled.jsonl --out-labels-pkl runs/new_data/frag_labels.pkl
```

The fragment extractor emits rows in molecule order. It uses capped, canonical BRICS fragments with at least five heavy atoms and at most six fragments per molecule. Select sequence length from the sanity report; archived runs used 84.

For molecule-based evaluation, use `python -m mvselfies.evaluate --help`. The historical main 10k/100k results used `--fragment-convention labelled`, whereas the 1M results used `capped`. The experiment runner records this distinction explicitly.

## License

The software, model weights, and original experimental results use the [MIT License](LICENSE). ChEMBL-derived datasets and training-reference data retain [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and their attribution requirements; see the [dataset README](datasets/README.md). Third-party dependencies retain their respective licenses.
