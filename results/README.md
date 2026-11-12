# Result bundle

Download [results.zip](https://zenodo.org/records/24147562/files/results.zip?download=1)
from [resource version 0.1.0 on Zenodo](https://doi.org/10.5281/zenodo.24147562).
Verify the archive against [SHA256SUMS.txt](https://zenodo.org/records/24147562/files/SHA256SUMS.txt?download=1).

Use `python src/mvselfies/restore_resources.py results.zip` at the repository root
to extract the result bundle so its paths begin with
`results/main/` and `results/replicates/`.
`manifest.json` records the resource file sizes and SHA-256 checksums.

`summary.csv` and `tables/` are included in Git.

- `main/`: the nine Baseline-VAE/Contrastive-VAE/Rare-Aware-VAE runs whose metrics match the main manuscript table.
- `replicates/`: nine later 1M runs, labelled seeds 1, 10, and 100, for the models
  discussed in the manuscript's run-variation analysis.
- `tables/`: original CSVs used for manuscript figures and the rare-count table.

Each run contains its valid generated canonical SMILES, molecular-property CSV,
training log, original library-version record and input hashes. Fragment results
are under `frag_eval_smiles/`, including the original report and train/gen count
tables. Reconstructible merged count tables and training sample dumps are omitted.

These files retain historical measurements. The summary reports each run
individually; it does not average runs with different epochs, software versions,
fragment conventions, or repeated seed labels. Contrastive-VAE's low scores are retained
because they are experimental results reported in the manuscript.

```console
python -m mvselfies.summarize --verify --out runs/verified_summary.csv
```

The command recomputes JSD, rare recall, and top-k mass from saved counts and checks
them against every original report using the saved fragment counts. Experiment
commands are documented in the [root README](../README.md#run-recorded-experiments).

Main 10k/100k evaluations retain labelled BRICS attachment atoms, while 1M
evaluations use hydrogen-capped fragments. Historical restricted-vocabulary
reports mix SELFIES vocabulary entries with SMILES counts and report 100% OOV;
these restricted scores do not measure fragment-distribution fidelity.
Fragmentation timeouts can also change evaluated molecule and fragment counts
between runs. Molecular-property analysis reports the `sa_proxy` heuristic
rather than the Ertl synthetic-accessibility score.

Original experimental results are licensed under the [MIT License](../LICENSE).
ChEMBL-derived training-reference data retain the upstream CC BY-SA 3.0 terms
described in the [dataset README](../datasets/README.md).
