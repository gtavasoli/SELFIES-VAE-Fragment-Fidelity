# Dataset bundle

Download [datasets.zip](https://zenodo.org/records/24147562/files/datasets.zip?download=1)
from [resource version 0.1.0 on Zenodo](https://doi.org/10.5281/zenodo.24147562).
Verify the archive against [SHA256SUMS.txt](https://zenodo.org/records/24147562/files/SHA256SUMS.txt?download=1).

Use `python src/mvselfies/restore_resources.py datasets.zip` at the repository root
to extract the dataset bundle so its paths begin with
`datasets/10k/`, `datasets/100k/`, and `datasets/1m/`.
Large data files are excluded from Git. `manifest.json` records their
sizes and SHA-256 checksums; `inventory.json` records measured row counts.

Each of `10k/`, `100k/`, and `1m/` contains:

| File | Meaning |
| --- | --- |
| `clean_smiles.txt` | Canonical molecular SMILES, one per row |
| `clean_selfies.txt` | Molecular SELFIES used in archived training |
| `selfies_vocab.txt` | Ordered vocabulary, including four special tokens |
| `frags_filled.jsonl` | Original fragment records, with molecule identities |
| `frag_vocab.txt` | Ordered fragment SELFIES vocabulary |
| `frag_freq.csv` | Fragment frequency counts |
| `frag_labels.pkl` | Fragment-vocabulary indices for each JSONL row |
| `sanity_report.json` | Historical preprocessing report |
| `token_freq.csv`, `excluded_log.csv` | Historical preprocessing diagnostics |

Measured molecular row counts are **9,995**, **99,748**, and **982,458**.
Scale names are nominal. The saved 1M sanity report predates the final data file
and reports a larger count. The measured inventory and archived input hashes
identify the released files.

The JSONL rows and pickle labels agree with each other, but almost all JSONL
molecule identities differ from the molecular SELFIES at the same position.
These original bytes are retained to preserve the historical experiment inputs.
Identity checks found 9,994 mismatches out of 9,995 rows at 10k, 99,690 out of
99,748 at 100k, and 982,376 out of 982,458 at 1M. Realigning these rows defines
a new experiment and requires retraining. The
[recorded experiment runner](../README.md#run-recorded-experiments) explicitly
enables the historical positional-pairing override.

Available files do not establish the manuscript's sampling seed or subset nesting.

The intermediate `frags.jsonl` has been omitted. It contains the same fragment
sets and molecule metadata as `frags_filled.jsonl`, with different fragment-list
ordering. Training uses the filled records listed above; the intermediate file
is not required.

The data originate from [ChEMBL](https://www.ebi.ac.uk/chembl/), release 35.
ChEMBL data are provided under
[Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/),
as described in the [ChEMBL licensing documentation](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions).
Preserve this attribution and the upstream terms when redistributing the data;
the software's MIT license does not replace them.
