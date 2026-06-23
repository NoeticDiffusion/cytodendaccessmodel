# E022R — Shuffled Replay Scaling Audit: Claim Ledger
## Branch counts tested: [4, 8, 16, 32]
## Seeds per condition: 20

## Monotonic decay of ratio_to_full_model?
- canonical: YES (confirmed)
- strong_overlap: YES (confirmed)

## Does shuffled_replay ever pass full joint profile at n >= 8?
- NO — sampling artifact confirmed

## Allowed claim

> Shuffled replay can partially mimic overlap-branch writing in the smallest
> four-branch motif because random reassignment has a high chance of revisiting
> the same branch (probability 1/n_branches per allocation weight).  This
> apparent match decays as branch allocation space increases, supporting the
> interpretation that identity-preserving replay, not replay alone, is required
> for scalable structural specificity.
