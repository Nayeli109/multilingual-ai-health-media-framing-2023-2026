# Multilingual Media Discourse on Generative AI, Health, Misinformation, Trust, and Regulation (2023–2026)

This repository contains the reproducibility package for a multilingual computational media-framing study of generative AI discourse in health, misinformation, trust, and regulation. The public release is intentionally separated from earlier exploratory/rejected-methodology material and preserves the final redesigned analysis pipeline, frozen protocols, selected aggregate results, and provenance records.

## Study design

The supervised component uses three tasks:

| Task | Classes |
|---|---:|
| `primary_frame` | 9 |
| `stance` | 5 |
| `misinformation_relation` | 4 |

Development analyses use the frozen `LegacyAux-199` benchmark with 4 outer folds and 5 seeds (`11, 29, 47, 83, 131`). The final prospective benchmark, `EventGold35`, contains 35 documents and was kept separate from development, architecture selection, epoch selection, and human-label adjudication.

### Modeling sequence

1. Corpus audit and leakage-aware holdout construction.
2. Inductive domain-adaptive pretraining (DAPT) variants.
3. Classical TF-IDF baselines.
4. BASE-vs-DAPT supervised comparison.
5. **Stage A: NO_EVENT XLM-R** development model.
6. Stage B verified same-event context with a parameter-matched `PERMUTED_CONTEXT` control.
7. V4 architecture-sensitivity analysis (FiLM and low-rank bilinear variants).
8. Final Stage-A refit on the frozen architecture and one-shot evaluation on `EventGold35`.
9. Post-hoc representation topology and generalization-gap diagnostics.

**Important:** Stage A is a **NO_EVENT** model. It must not be described as event-aware.

## Final architecture decision

The frozen final architecture is `STAGE_A_NO_EVENT`. Stage B and V4 analyses were used to test whether verified event context or alternative fusion architectures yielded confirmatory gains. They did not satisfy the predeclared selection rule, so no contextual architecture replaced Stage A before the held-out evaluation.

## Held-out EventGold35 performance

The final five-seed Stage-A refit ensemble was evaluated once on the frozen 35-document EventGold benchmark.

| Task | Macro-F1 | 95% bootstrap CI | Balanced accuracy | Accuracy |
|---|---:|---:|---:|---:|
| `primary_frame` | 0.2761 | [0.1957, 0.3561] | 0.5684 | 0.4000 |
| `stance` | 0.4393 | [0.3145, 0.6018] | 0.6497 | 0.4286 |
| `misinformation_relation` | 0.5076 | [0.4213, 0.5929] | 0.5159 | 0.6000 |

The held-out benchmark is reported separately from development performance. No EventGold metric was used for architecture or epoch selection.

## Development findings

On `LegacyAux-199`, Stage A improved Macro-F1 relative to the frozen XLM-R BASE comparator for all three tasks:

| Task | BASE Macro-F1 | Stage A Macro-F1 | Delta |
|---|---:|---:|---:|
| `primary_frame` | 0.3394 | 0.4539 | +0.1145 |
| `stance` | 0.4172 | 0.4698 | +0.0526 |
| `misinformation_relation` | 0.4673 | 0.5694 | +0.1020 |

Only the primary-frame contrast met the predeclared multiplicity-adjusted support criterion. Null results are not interpreted as evidence of equivalence.

For Stage B, verified-context gains were not confirmatory after the prespecified inference procedure and multiplicity control. The `PERMUTED_CONTEXT` arm is a specificity/negative control and is not treated as a causal intervention.

## Representation analyses

Post-hoc representation analyses are exploratory and mechanistic rather than confirmatory. They show that DAPT preserves BASE geometry very strongly at the global level (including CKA ≈ 0.9996 and high neighborhood retention), while task-label geometry remains comparatively weak. Spectral and persistent-homology summaries indicate small reorganization rather than a wholesale representational transformation.

The language/media-system axis is strongly encoded in representation space, but fold-safe axis removal did not yield a performance-gap recovery that was clearly exceptional relative to permutation-derived control axes. These analyses must not be interpreted as causal mediation.

## Generalization diagnostic

A post-hoc development-to-EventGold comparison quantifies external performance differences and label-composition differences. Because the development and EventGold samples are distinct, the analysis uses independent within-dataset stratified bootstrap summaries rather than paired hypothesis tests. Composition differences are descriptive and are not treated as proof of distribution shift.

## Repository structure

```text
.
├── src/          # Final public analysis/training/evaluation scripts
├── protocols/    # Frozen protocols, analysis plans, locks, and closures
├── results/      # Aggregate public results and selected supplements
├── provenance/   # Public-release manifest, content audit, and frozen hashes
├── .gitattributes
├── .gitignore
├── README.md
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

The scientific package contains **155 manifest-tracked artifacts**:

- 60 code files
- 34 protocol/lock/closure files
- 43 primary result files
- 17 supplementary result files
- 1 frozen DAPT provenance artifact

The repository-level metadata files (`README.md`, `CITATION.cff`, `LICENSE`, and `requirements.txt`) are intentionally outside that frozen scientific-artifact count.

## Data availability and redistribution limits

Raw news/article text, private annotation keys, blind annotation packets, private masters, full EventGold source text, per-document prediction exports, model checkpoints, embeddings, and large intermediate arrays are **not redistributed in this public repository**.

This is deliberate. Some source texts are subject to copyright or redistribution constraints, and some annotation artifacts were kept private to preserve the integrity of the human-validation workflow. Public aggregate metrics, protocols, code, quality-control summaries, and provenance hashes are provided instead.

The absence of raw source text means that some end-to-end reproduction steps require access to the original licensed/collected source material. The repository is therefore a **transparent computational reproducibility package**, not a redistribution of the underlying media corpus.

## Human validation

`EventGold35` was dual-annotated under a model-blind procedure. After exact-text quality control, pre-adjudication agreement was:

- `primary_frame`: 32/35 agreement, Cohen's κ = 0.8890
- `stance`: 33/35 agreement, Cohen's κ = 0.9092
- `misinformation_relation`: 33/35 agreement, Cohen's κ = 0.8968

Seven task-level disagreements were adjudicated. Model predictions, probabilities, logits, and final EventGold performance were not used during adjudication.

## Environment

Reference runtime:

- Python 3.11.14
- PyTorch 2.6.0
- Transformers 4.57.6
- NumPy 1.26.4
- pandas 2.2.3
- scikit-learn 1.5.2
- SciPy 1.16.3
- sentence-transformers 3.0.1
- NetworkX 3.6.1
- ripser 0.6.12 for the frozen topology analysis

See `requirements.txt` for the pinned public environment.

The included `.sbatch` files assume a SLURM-based HPC environment. Cluster-specific paths, partitions, and scheduler directives may require adaptation on another system.

## Integrity and provenance

The frozen public-release manifest is:

```text
provenance/PUBLIC_RELEASE_MANIFEST_v1.tsv
SHA-256: 251346533c7b198aa4140e6dd34057c9a9887b00e7be80381827e84090ed274f
```

The public-content audit is:

```text
provenance/PUBLIC_RELEASE_CONTENT_AUDIT_v1.tsv
SHA-256: 18840ffa542a3c495f1eee2040daaebfb6d3a6d050b775849267692d5dd3afba
```

At packaging time, all 155 manifest destinations were present and matched their recorded SHA-256 values byte-for-byte.

## Reproducibility boundaries

When using this repository, keep the following distinctions explicit:

- Development (`LegacyAux-199`) is not final validation.
- Stage A is `NO_EVENT`.
- Folds and random seeds are repeated evaluations, not independent article-level sample sizes.
- Non-significant contrasts are not evidence of equivalence.
- The event verifier is not claimed to be calibrated.
- `PERMUTED_CONTEXT` is not a causal control.
- EventGold was not used for model selection.
- Post-hoc topology, nuisance-axis, and generalization analyses are exploratory.
- Label-composition differences alone do not establish distribution shift.
- Predictive entropy/max-probability summaries are not calibration metrics.

## Citation

Please cite this software/reproducibility repository using `CITATION.cff`. A manuscript citation can be added once the associated article is formally published.

## License

Code and repository-authored documentation are released under the MIT License. This license does **not** grant redistribution rights for third-party news/article content that is intentionally excluded from the repository.
