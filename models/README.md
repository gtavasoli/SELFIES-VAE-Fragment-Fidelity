# Trained model bundle

Download [models.zip](https://zenodo.org/records/24147562/files/models.zip?download=1)
from [resource version 0.1.0 on Zenodo](https://doi.org/10.5281/zenodo.24147562).
Verify the archive against [SHA256SUMS.txt](https://zenodo.org/records/24147562/files/SHA256SUMS.txt?download=1).

Use `python src/mvselfies/restore_resources.py models.zip` at the repository root
to extract the trained model bundle so its paths begin with
`models/main/` and `models/replicates/`.
`manifest.json` records every model's size and SHA-256 checksum.

Paths follow `models/<group>/<run>/best.pt`, for example:

```text
models/main/Rare-Aware-VAE_1m_s1/best.pt
models/replicates/Rare-Aware-VAE_1m_s10/best.pt
```

There are 18 generative checkpoints: nine matching the main table and nine
additional 1M runs. Their release folders use the manuscript names Baseline-VAE,
Contrastive-VAE, and Rare-Aware-VAE.
The [experiment catalog](../experiments/runs.json) links each model to its results and source
archive, with original configuration and checkpoint hashes.

Archived metadata retain the identifiers M0, M2, and M2pp (M2++), corresponding
to Baseline-VAE, Contrastive-VAE, and Rare-Aware-VAE, respectively. Source archive
paths identify the original exports; the released resources are sufficient to
run the package.

Each exported checkpoint contains the original VAE `model_state`, original
training arguments, saved optimizer step, ordered SELFIES vocabulary, and the
SHA-256 hash of its source checkpoint. Every model tensor is preserved. Optimizer
state is excluded from the inference exports.

These files support generative inference. Contrastive-VAE and Rare-Aware-VAE
checkpoints omit the auxiliary fragment encoder and BoW head, preventing complete
multiview training resumption. Training those models starts from scratch.

Run commands from the repository root. The vocabulary and architecture are loaded automatically:

```console
python -m mvselfies.generate --ckpt models/main/Rare-Aware-VAE_1m_s1/best.pt --out runs/demo/generated.smi --n 100
```

## Recorded training settings

InfoNCE receives `mu.detach()` by default. With Contrastive-VAE's mean pooling,
this loss does not directly update the molecular encoder. Rare-Aware-VAE retains
an indirect gradient path through its undetached attention query.
`--symmetric-info` removes the detach but does not add a reverse-direction loss.

Fragment frequency ranks increase from common to rare, and dropout is
proportional to `rank ** alpha`. Rare fragments have higher base dropout
probabilities, with a separate rule restoring at least one rare fragment when
available. This differs from the manuscript's description of preferentially
dropping common fragments.

Baseline-VAE uses a fixed KL coefficient of one. The multiview variants warm it
up over 100,000 steps. Train/validation splits use each run's training seed
rather than the manuscript's separate fixed seed 13455. Recorded epoch counts
are listed in the [experiment catalog](../experiments/runs.json).

Several original Baseline-VAE logs contain only the device line; missing learning
curves have not been reconstructed.

The model weights are licensed under the [MIT License](../LICENSE).
