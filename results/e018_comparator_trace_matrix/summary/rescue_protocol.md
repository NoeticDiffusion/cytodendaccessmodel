# e018 — Rescue Protocol

## Targeted overlap rescue
- Apply B1_CUE (only b1 driven) x 3 reps
- Run consolidation (replay mu1+mu2) x 3 passes
- Repeat 3 rounds
- Total consolidation passes: 9

Rationale: explicitly rebuilds E_b on b1 before each consolidation window,
targeted at the damaged overlap branch.

## Generic rescue (no pre-cueing baseline)
- Run plain consolidation (replay mu1+mu2) x 9 passes
- No targeted pre-cueing of any branch

Rationale: same total consolidation volume as targeted rescue, but without
branch-specific E_b pre-loading. Matches the 'standard rescue' comparison
used in exp015. This ensures SIG-E tests whether b1-specific pre-cueing
confers additional advantage over consolidation alone.

Note: an earlier version used uniform all-branch pre-cueing (GENERIC_CUE) as the
baseline, but this also drove b1 at 0.5, making it too similar to targeted rescue.
Plain consolidation (no pre-cueing) is the more meaningful control.

## SIG-E definition
```
SIG_E = recovery_pct(L_targeted) - recovery_pct(L_generic)
recovery_pct = (L_post_rescue - L_post_damage) / (L_healthy - L_post_damage) x 100
```

## Interpretation
SIG-E > 0 means targeted overlap rescue recovers more linking than plain re-consolidation.
SIG-E > 10.0 pp = protected threshold.
