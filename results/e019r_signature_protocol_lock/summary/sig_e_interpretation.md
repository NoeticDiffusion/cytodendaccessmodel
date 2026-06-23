# SIG-E — Interpretation and Unit Discipline

## Canonical values (E019R / E019 protocol)

| Quantity | Value |
|---|---|
| L_pre_damage (healthy) | 0.6461 |
| L_post_damage | 0.5047 |
| damage = L_healthy - L_post_damage | 0.1414 |
| NR_targeted | 0.9947 |
| NR_generic_plain | -0.5526 |
| SIG_E_normalized = NR_targ - NR_gen | 1.5473 |
| SIG_E_raw = L_targ - L_gen | 0.2188 |
| Overshoot (targeted) | False |

## Unit decision
SIG-E_normalized CAN exceed 1.0 (targeted rescue overshoots healthy baseline).
Label is "normalized recovery difference", NOT "percentage points".

## Rescue protocol sensitivity
SIG-E magnitude depends critically on whether pre-rescue probe cues are present:
- E017/E018 (probe cues present): SIG-E ≈ 0.33 NR units (old pp = 33)
- E019/E019R (no probe cues):    SIG-E ≈ 1.5473 NR units

Canonical forward protocol = E019 variant (no inter-phase probes).
