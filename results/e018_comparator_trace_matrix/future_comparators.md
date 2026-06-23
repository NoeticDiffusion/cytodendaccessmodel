# e018 — Future Comparators (not implemented in E018)

The following five optional comparators from the E018 spec were not implemented
in this run to avoid delaying the core five-comparator result. They are documented
here for E018R or E019.

## hebbian_weight_only
Associative strengthening via trace weight updates, not M_b.
Implementation: add a per-trace weight matrix that scales recall support; disable M_b writing.

## soma_global_gain_only
A global gain G(t) applied uniformly to all branches.
Implementation: multiply all branch activations by a shared scalar; disable M_b.

## shuffled_replay
Replay exists but branch identity is shuffled before consolidation.
Implementation: permute branch_id → M_b assignment in each consolidation pass.

## eligibility_only
E_b affects recall transiently but no persistent M_b is written.
Implementation: structural_lr=0; let E_b directly scale recall support via a fast gain.

## resource_only
P_b (capture/resource) exists and accumulates but no persistent M_b is retained.
Implementation: structural_lr=0; let P_b directly scale recall support.

These comparators would strengthen the model-discrimination claim if they also
fail the joint SIG-A–SIG-E profile.
