# e017 — Effect Summary

## Protocol
Canonical 4-branch simulator: b0, b1 (overlap), b2, b3.
Two traces: mu1 (b0/b1-dominant), mu2 (b1/b2-dominant).
10 phases: init → encode → probe → consolidation → probe →
           damage → probe → rescue → probe.

## Parameters
structural_lr=0.18, replay_gain=0.8,
eligibility_decay=0.12,
structural_gain=6.0,
consolidation_passes=9

## Key values

| Metric | Value |
|--------|-------|
| M_b1 pre-consolidation  | 0.5000 |
| M_b1 post-consolidation | 0.7986 |
| ΔM_b1 (overlap branch)  | +0.2986 |
| ΔM_b mean (non-overlap) | +0.1705 |
| L pre-consolidation     | 0.4075 |
| L post-consolidation    | 0.6472 |
| L post-damage           | 0.5058 |
| L standard rescue       | 0.6081 |
| L targeted rescue       | 0.6552 |
| R_mu1 post-cons         | 1.4237 |
| R_mu1 post-damage       | 1.4112 |
| Link drop %             | +21.9% |
| Recall drop %           | +0.9% |
| Standard rescue %       | +72.4% |
| Targeted rescue %       | +105.7% |

## Protected signatures

| Signature | Score | Direction |
|-----------|-------|-----------|
| SIG-A overlap advantage     | 0.1281 | PASS |
| SIG-B linking gain          | 0.2397 | PASS |
| SIG-C context separation    | 0.1545 | PASS |
| SIG-D dissociation (pp)     | 20.98pp | PASS |
| SIG-E rescue advantage (pp) | 33.29pp | PASS |
