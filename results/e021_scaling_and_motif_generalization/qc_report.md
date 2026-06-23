# E021 — QC Report

## Signature computation
gSIG-A to gSIG-E computed inline from protocol outputs.
gSIG-C uses private-cue recall probe (no separate context simulation).
gSIG-C is architectural — NOT required for joint_pass.

## Joint pass criterion
joint_pass = gSIG-A > 0 AND gSIG-B > 0 AND gSIG-D > 0 AND gSIG-E > 0

## False-linking rate
false_linking_rate = delta_L_unlinked / max(delta_L_expected, eps)
NaN when no unlinked pairs defined (hub_overlap).

## SIG-E unit
gSIG-E = NR_targeted - NR_generic (normalized recovery difference, NOT pp)

## Protocol
encode(2x per trace) -> consolidate(9) -> damage(9 null) ->
targeted_rescue(3 rounds: 3 cue reps + 3 passes) -> generic_rescue(9 passes)
