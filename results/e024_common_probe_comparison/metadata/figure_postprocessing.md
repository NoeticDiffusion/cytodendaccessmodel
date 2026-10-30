# E024 figure post-processing note

The original Swedish `Fig_e024_01_primary_effects.png` remains unchanged. The article
uses the English-language rendering `figures/Fig_e024_01_primary_effects_en.png`, copied
to the manuscript's `figures2/Fig_e024_01_common_probe_comparison.png`.

The figure-only renderer reads `summary/primary_model_summary.csv`; it does not recompute
the experiment, alter any observation, or change any statistical summary.

| Artifact | SHA-256 |
|---|---|
| Input `summary/primary_model_summary.csv` | `28e3313a17d4d3e2258d681e1ddefbdd0321729464d9f46f38594408eb5df49f` |
| Renderer `experiments/exp024_render_english_figure.py` | `3d4ab3fc05f2645b2ff47fca5d41f436317355e2c565925c6fb0f0f376410781` |
| English result figure | `caaf6ed68777b564d125f8d1cd343ae6574c2e0776abcf88bc509b47e169cb85` |

The frozen E024 run configuration and evaluated experiment-script hash are preserved
separately in `metadata/run_config.json`.

## Registered recovery-arm aggregation

`experiments/exp024_summarize_secondary.py` pairs targeted and private-control rows,
averages eligible pairs within motif and motifs within seed, and uses a descriptive
10,000-resample seed-cluster bootstrap. It does not alter `secondary_by_seed.csv` or add
a multiplicity-adjusted cross-model test.

| Artifact | SHA-256 |
|---|---|
| Input `summary/secondary_by_seed.csv` | `b6b9d7b659fa1d8bc7f4052e0e31a3eb7bd91c0dc8f8d868ae5463bb8f4651d3` |
| Aggregator `experiments/exp024_summarize_secondary.py` | `527844c036b21a021e8902bcd832098691cf3f4e7f250b71c2bbdbc2e4ad9c3c` |
| `summary/recovery_seed_summary.csv` | `a3b9bcd8f2dcf7e922d63d52b44814289c0ed96a20b9557eaf58fb471491cac4` |
| `summary/recovery_model_summary.csv` | `fe56a9da63d9c4c686d32cdc98c94b30d3316eb312e13de34268dd76b1f3b937` |
