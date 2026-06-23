# E021R — Article-Facing Motif Language

## Canonical statement

> The slow-writing mechanism generalized beyond the canonical four-branch/two-trace case
> in motifs with sufficient but not universal overlap. Strong-overlap, chain, and
> sparse-random motifs preserved the expected structural writing and linking signatures,
> while weak-overlap motifs failed as predicted by the overlap-threshold boundary
> identified in E020. Hub-overlap motifs produced strong structural writing and linking
> but lacked specificity because all traces shared the same hub branch. We therefore treat
> hub overlap as an over-linking boundary condition rather than as a clean success case.

## Per-class wording

### canonical_reference
> Canonical four-branch behavior reproduced at branch counts 4, 8, 16, and 32.
> gSIG-A through gSIG-E are stable, confirming that scaling the number of neutral
> branches does not degrade the mechanism.

### specific_linking_success (strong_overlap)
> Strong overlap (weight 0.95) produces stronger linking gain (gSIG-B = +0.295 vs
> canonical +0.239) while maintaining rescue selectivity. No false-linking concerns
> (only two traces; unlinked pairs undefined).

### chain_local_linking_with_leakage (chain_overlap)
> Chain topology shows that local adjacent-pair linking (t0–t1, t1–t2) is preserved
> while distant pairs (t0–t2) gain only ~19% as much linking (FL ≈ 0.19).
> Local specificity is robust; some cross-chain contamination occurs through the
> intermediate trace. This is a valid partial-specificity finding.

### hub_overlinking_boundary (hub_overlap)
> Hub topology produces strong structural writing (gSIG-A > 0) and linking gain
> (gSIG-B > 0), but all trace pairs become universally linked through the shared
> hub branch. False-linking specificity is undefined (no unlinked pairs).
> **Hub motifs should not be counted as clean generalization successes.**
> The correct interpretation is: hub topology is a boundary condition at which the
> mechanism produces over-linking rather than selective linking.

### sparse_specific_success (sparse_random)
> Sparse random allocation with density 0.30 produces meaningful linking for pairs
> with realized overlap while maintaining partial specificity (FL = 0.16–0.30,
> moderate band). Results are consistent across two random seeds and three branch counts.

### weak_overlap_failure (weak_overlap)
> Overlap weight 0.30 falls below the threshold identified in E020 (≈ 0.40).
> The overlap branch cannot accumulate sufficient structural accessibility, and
> gSIG-A becomes negative (non-overlap branches outperform the overlap branch).
> This is a predicted, mechanistically interpretable failure, not a model defect.

## Implications for manuscript

Allowed claims after E021R:
- The slow-writing mechanism generalizes to selected non-canonical motifs.
- The mechanism requires sufficient overlap weight (> threshold).
- Hub-like overlap produces strong linking but poor specificity.
- Chain and sparse motifs preserve partial specificity under tested conditions.
- The generalized result is bounded by overlap topology.

Not allowed:
- All motif classes are clean successes.
- Hub overlap proves generalization without qualification.
- The model generalizes to arbitrary memory systems.
- The model scales to realistic dendritic trees.
