#import "template.typ": essay-template

#show: doc => essay-template(
  short_title: [S3 Appendix],
  doc,
)

= S3 Appendix. Open-data pipelines and reproducibility details

== Scope

The open-data results in the main text are intended as downstream signature tests of a latent structural-accessibility hypothesis. They do not directly observe $M_b$ or any microscopic write variable. Their value lies instead in using fixed, inspectable pipelines to ask whether the predicted observable consequences survive in real neural recordings.

== Public code and data record

All author-generated code, scripts, configuration files, and manuscript-facing outputs associated with these pipelines are publicly available in the `cytodendaccessmodel` GitHub repository (`https://github.com/NoeticDiffusion/cytodendaccessmodel`) and in the manuscript-matched Zenodo snapshot (DOI `10.5281/zenodo.19498499`; `https://doi.org/10.5281/zenodo.19498499`). The code is released under the GNU General Public License v3.0 (`GPL-3.0`). Third-party neural data are not redistributed in this repository; they remain accessible from their original DANDI Archive records (`000718`, `000336`, `001710`), while the derived manuscript-facing outputs discussed here are written under `data/dandi/triage/000718`, `data/dandi/triage/000336`, and `data/dandi/triage/001710`.

== DANDI `000718`: Offline core-unit enrichment pipeline

The `000718` analysis targets a narrow question: are the units most central to a NeutralExposure ensemble preferentially reactivated during later high-synchrony offline events?

The retained pipeline is:

1.  Cross-session ROI registration between NeutralExposure and offline sessions, using confidence-filtered spatial matching to define candidate unit identities across days @Sheintuch2017CellRegistration @Vergara2025CaliAli.
2.  Ensemble extraction from the NeutralExposure session using NMF with `k = 8`, followed by definition of core units as the top `15%` of weights within each ensemble @Nagayama2022NMFAssemblies.
3.  Detection of offline high-synchrony events from the offline session using event-first population synchrony criteria, with event windows compared against duration-matched inter-event windows from the same session @NavasOlive2024RipplAI @Liu2022ECannula.
4.  Computation of a session-level enrichment statistic for each event-ensemble pair: active core-unit fraction during the event minus the average active fraction across ten matched inter-event windows.

Robustness checks include an activity-threshold sweep (`0.0`, `0.5`, `1.0 sigma`) and registration-shuffle controls. The result that survives these checks is modest but consistent: NeutralExposure-defined core units are somewhat more active during detected offline events than expected from matched inter-event baseline, yet much of the absolute signal remains attributable to the general population-burst character of those events.

#table(
  columns: (auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, center, center, left),
  table.header(
    [*Pair count*], [*Threshold sweep*], [*Shuffle control*], [*Retained claim*],
  ),
  [3 NeutralExposure-to-offline pairs], [positive enrichment at `0.0`, `0.5`, `1.0 sigma`], [real mappings exceed registration-shuffle baseline, but modestly], [offline core-unit enrichment is positive but sits above a strong generic burst background],
)

#par(first-line-indent: 0pt)[
  #emph[Table 7. Compact robustness summary for the DANDI `000718` offline core-unit enrichment analysis.]
]

== DANDI `000336`: Cross-plane access-constraint pipeline

The `000336` analysis targets a different question: do paired imaging planes communicate above null while remaining less coupled than units within the same plane?

The retained full-bundle pipeline is:

1.  Load all six NWB files and organize them as three within-session pairs: two primary cross-depth pairs and one supplementary cross-area pair.
2.  Separate spontaneous from stimulus-condition windows and use exact timestamp alignment when the two planes share acquisition timing; for the supplementary `ses-1245548523` cross-area session, use coarser `0.5 s` binning because timestamps are interleaved rather than shared.
3.  Merge short stimulus presentations into block-level windows before computing pairwise correlations, while retaining a conservative minimum block duration rule.
4.  Compare observed cross-plane coupling against within-plane coupling and against within-window circular-shift nulls (`n = 200`).

Under this conservative pipeline, spontaneous cross-plane coupling is above null in all three pairs (`z = 4.04` to `5.16`) and remains above null across all tested stimulus conditions. The stricter `cross < both within` criterion is fully met in the supplementary `ses-1245548523` cross-area pair, but only partially met in the two cross-depth pairs because one within-plane estimate is unusually weak or unusually strong. The retained claim is therefore bounded: the full bundle supports structured above-null inter-plane coupling across all analyzed bundle pairs and tested conditions, while only the supplementary cross-area pair supplies a clean bilateral access-constraint match.

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, center, center, center, left),
  table.header(
    [*Pair*], [*Geometry*], [*Spontaneous z*], [*All tested conditions above null?*], [*Strict `cross < both within` verdict*],
  ),
  [pair_a], [cross-depth], [4.98], [yes], [partial],
  [pair_b], [cross-depth], [4.04], [yes], [partial],
  [pair_c], [cross-area], [5.16], [yes], [positive],
)

#par(first-line-indent: 0pt)[
  #emph[Table 8. Compact robustness and claim-boundary summary for the DANDI `000336` full-bundle coupling analysis.]
]

== DANDI `001710`: Context-sensitive place-code and remapping baseline

The `001710` analysis targets a third question: do longitudinal hippocampal population codes show the combination of stable spatial tuning, within-day arm separation, and partial cross-day continuity expected from a context-sensitive accessibility account, and does perturbing a candidate postsynaptic structural-write mechanism reduce that continuity selectively across days?

The retained group-bundle pipeline is:

1.  Load the broad `23`-subject longitudinal bundle (`7` Cre, `9` Ctrl, `7` SparseKO; `139` NWB files total) and verify readiness with the generic `dandi_io` inventory and probe outputs.
2.  Generalize the I/O layer so that both single-channel and multi-channel sessions resolve behavior and ophys containers without hardcoded interface names.
3.  Reconstruct trials from the 2P-aligned behavior channels (`trial start`, `trial end`, `trial number`, `block`, `left or right`) cross-referenced against the embedded `trial_cell_data` annotation blob.
4.  Extract conservative `df` matrices together with aligned behavior, then compute occupancy-normalized tuning curves for each ROI separately within each session or channel.
5.  Compute subject-level observables: within-day left-versus-right arm population-vector structure, cross-day similarity from index-matched ROI tuning curves, subject-level permutation nulls for group comparisons, and genotype-aggregated day-lag profiles. For SparseKO subjects, the canonical group comparison uses `ch0` as a predefined single-channel summary so that each animal contributes one observation to the genotype contrast, with channel-level robustness summarized separately below.

Under this broad group-bundle pipeline, all `139` files were locally available and entered the QC sweep. Subject-level similarity summaries retained `23` subjects in total: `18` with full `6`-day coverage, `4` with `5` usable days, and `1` with `4` usable days. Split-half reliability remained high across groups, and the retained cross-day means were `0.3374` for Cre, `0.2926` for Ctrl, and `0.2623` for SparseKO. The broad retained claim is therefore stronger than in the original replication bundle: lower cross-day stabilization in SparseKO is now supported at the subject-group level rather than only by a single-animal bridge.

A dedicated follow-on robustness run (`experiments/dandi_001710_08_robustness_and_nulls.py`) added four checks. First, the arm-label audit found `vr_trial_info` content but not a clean per-trial arm truth table, so arm counts remain a QC sanity check rather than a direct label-validation result. Second, the subject-level group null tests placed SparseKO below Cre but not cleanly below Ctrl. Third, the lag profile remained lower for SparseKO across lags `1` to `5`, arguing against the effect being driven by a single interval. Fourth, SparseKO channel comparison showed quantitative sensitivity rather than a qualitative reversal: channel `1` averaged somewhat higher cross-day stability and slightly stronger arm separation than channel `0`, while both channels remained high-reliability. The main-text use of `ch0` should therefore be understood as a predefined bookkeeping choice for one-channel-per-animal comparison, not as evidence that `ch0` is uniquely privileged biologically.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, center, center, center, center, left),
  table.header(
    [*Group*], [*Subjects*], [*Usable days / subject*], [*ROI range*], [*Split-half reliability*], [*QC / interpretive note*],
  ),
  [Cre], [7], [5-6], [250-1840], [0.949-0.986], [highest stability mean; clean reference group],
  [Ctrl], [9], [4-6], [639-1694], [0.817-0.984], [broader heterogeneity but still usable baseline],
  [SparseKO], [7], [5-6], [ch0: 140-504; ch1: 31-199], [ch0: 0.884-0.980; ch1: 0.951-0.992], [lowest canonical stability mean; channel-sensitive rather than channel-broken],
)

#par(first-line-indent: 0pt)[
  #emph[Table 9. Compact QC summary for the broadened DANDI `001710` genotype bundle, highlighting group coverage, ROI ranges, and split-half reliability before claim interpretation.]
]

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, center, center, center, center, center, left),
  table.header(
    [*Comparison*], [*SparseKO mean*], [*Other mean*], [*SparseKO n*], [*Other n*], [*z / p*], [*Claim*],
  ),
  [SparseKO vs Cre], [0.2623], [0.3374], [7], [7], [`-2.1495 / 0.009`], [SparseKO lower than Cre; null-separated],
  [SparseKO vs Ctrl], [0.2623], [0.2926], [7], [9], [`-1.2856 / 0.099`], [directionally lower, but not null-separated],
)

#par(first-line-indent: 0pt)[
  #emph[Table 10. Subject-level permutation null tests for the broadened DANDI `001710` genotype comparison. SparseKO falls below Cre under the implemented subject-level null and is directionally lower than Ctrl, but the latter contrast is weaker.]
]

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, center, center, center, left),
  table.header(
    [*Metric*], [*ch0*], [*ch1*], [*ch0 - ch1*], [*Interpretation*],
  ),
  [Cross-day similarity], [0.2623], [0.3296], [-0.0673], [channel `1` is somewhat more stable, but the effect is not reversed],
  [Within-day arm separation], [0.2796], [0.3119], [-0.0323], [both channels preserve within-day structure; `ch1` is slightly stronger on average],
  [Split-half reliability], [0.9602], [0.9767], [-0.0165], [both channels remain high-reliability],
)

#par(first-line-indent: 0pt)[
  #emph[Table 11. SparseKO channel comparison for the broadened DANDI `001710` robustness pass. The two channels differ quantitatively but not qualitatively: both support preserved within-session structure, while channel `1` is somewhat more stable across days on average.]
]

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, center, center, center, center, center),
  table.header(
    [*Cohort*], [*lag1*], [*lag2*], [*lag3*], [*lag4*], [*lag5*],
  ),
  [Cre], [0.3445], [0.3299], [0.3307], [0.3265], [0.3634],
  [Ctrl], [0.2992], [0.2931], [0.2924], [0.2819], [0.2785],
  [SparseKO], [0.2812], [0.2766], [0.2555], [0.2340], [0.1973],
)

#par(first-line-indent: 0pt)[
  #emph[Table 12. Group-level day-lag profile for the broadened DANDI `001710` robustness pass. SparseKO remains lower than the non-KO cohorts across all tested day-lags, arguing against the stability deficit being driven by a single recording interval.]
]

== Repository entry points

The public GitHub repository and archived Zenodo snapshot are structured so that reviewers can inspect the exact scripts and configs used for these observable bridges. The main entry points are:

- `configs/dandi/dataset_000718.yaml`
- `configs/dandi/dataset_000336.yaml`
- `configs/dandi/dataset_001710.yaml`
- `configs/dandi/dataset_001710_replication_bundle_01.yaml`
- `configs/dandi/dataset_001710_group_bundle_01.yaml`
- `python -m dandi_io.cli list --config <yaml>`
- `python -m dandi_io.cli download --config <yaml>`
- `python -m dandi_io.cli probe --config <yaml>`
- `experiments/dandi_000718_14_h1_pri_enrichment.py`
- `experiments/dandi_000336_04_sub656228_replication.py`
- `experiments/dandi_000336_05_ses1245548523.py`
- `experiments/dandi_000336_06_full_bundle.py`
- `experiments/dandi_001710_03_trial_reconstruction.py`
- `experiments/dandi_001710_05_within_day_place_tuning.py`
- `experiments/dandi_001710_06_cross_day_remapping_baseline.py`
- `experiments/dandi_001710_07_replication_bundle_full_pass.py`
- `experiments/dandi_001710_08_robustness_and_nulls.py`

The resulting artifacts are written under `data/dandi/triage/000718`, `data/dandi/triage/000336`, and `data/dandi/triage/001710`, including the `000336` full-bundle coupling outputs, the `001710` replication-bundle and group-bundle outputs, and the dedicated `001710/robustness/` outputs for null tests, QC summaries, day-lag profiles, and channel comparisons. This makes the empirical claims in the manuscript inspectable at the level of scripts, configs, and output files rather than only at the level of prose summary.

#bibliography("references_cytoskeletal_dendritic_accesibility_model.bib")
