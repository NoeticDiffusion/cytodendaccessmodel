# Signature Definitions (locked at E019R)

## SIG-A — Overlap-branch selective structural writing

**Formula:**
```
SIG_A = (M_b(overlap, post) - M_b(overlap, pre)) - mean(M_b(non-overlap, post) - M_b(non-overlap, pre))
```
**Units:** ΔM_b (dimensionless, approx [-1, 1])
**Bounded:** Yes (M_b ∈ [0, 1])
**Directional threshold:** > 0
**Protected threshold:** > 0.02
**Diagnostic:** Primary — requires slow structural write; fails when structural_lr = 0
**Non-diagnostic cases:** None identified

---

## SIG-B — Linking gain after consolidation

**Formula:**
```
SIG_B = L_post_consolidation - L_pre_consolidation
L = Σ_b a_{μ1,b} · a_{μ2,b} · M_b
```
**Units:** ΔL (dimensionless, bounded by allocation geometry)
**Bounded:** Yes (L ∈ [0, max_alloc²])
**Directional threshold:** > 0
**Protected threshold:** > 0.05
**Diagnostic:** Primary — requires slow structural write; fails when structural_lr = 0 or replay_gain = 0
**Non-diagnostic cases:** None identified

---

## SIG-C — Context separation

**Formula:**
```
SIG_C = mean( (r_corr_α - r_wrong_α) + (r_corr_β - r_wrong_β) ) / 2
```
**Units:** recall-support units (dimensionless)
**Bounded:** Approximately (support values bounded by readout_gain and activation)
**Directional threshold:** > 0
**Protected threshold:** > 0.05
**Diagnostic:** ARCHITECTURAL / AUXILIARY — reflects allocation geometry (which
branches mu_alpha and mu_beta occupy). SIG-C passes even when structural_lr = 0 and
context_gain = 0. It does NOT diagnose slow structural writing.
**E018 finding:** All 5 comparators passed SIG-C (including fast_context_only,
fixed_allocation_only). Allocation structure is sufficient.
**E019 finding:** All 7 context_gain values (0.00–3.00) pass SIG-C.
**Article implication:** Report SIG-C as an architectural fast-gating signature,
not as a discriminative slow-writing marker.
**Non-diagnostic cases:** Comparators without slow writing (fast_context_only, etc.)

---

## SIG-D — Linking > recall dissociation

**Formula:**
```
SIG_D = (L_drop_pct) - (recall_drop_pct)
      = 100 × (L_post - L_dmg)/L_post - 100 × (recall_post - recall_dmg)/recall_post
```
**Units:** percentage points (both input ratios are bounded)
**Bounded:** Yes (each drop_pct ∈ [-∞, 100])
**Directional threshold:** > 0
**Protected threshold:** > 5.0 pp
**Diagnostic:** PERTURBATION-SENSITIVE / NON-SPECIFIC — SIG-D passes when the damage
specifically targets the overlap branch (geometry effect) regardless of whether M_b
was dynamically written. Passes in E018 for fast_context_only, replay_no_structure,
and fixed_allocation_only. Not diagnostic alone.
**Article implication:** SIG-D is meaningful only in combination with SIG-A and SIG-B.
**Non-diagnostic cases:** All comparators with structural overlap geometry

---

## SIG-E — Targeted rescue selectivity

**Formula (canonical, locked at E019R):**
```
SIG_E_raw        = L_targeted_rescue - L_generic_plain_consolidation_rescue
SIG_E_normalized = NR_targeted - NR_generic_plain
NR               = (L_post_rescue - L_post_damage) / (L_healthy - L_post_damage)
```
**Units:**
  SIG_E_raw:        ΔL (dimensionless, absolute difference)
  SIG_E_normalized: dimensionless NR difference (CAN exceed 1.0 = overshoot)
**DO NOT label SIG-E in "percentage points"** — NR is unbounded above 1.0
**Protected threshold:** SIG_E_normalized > 0.10 OR SIG_E_raw > 0.02
**Protocol sensitivity:** PROTOCOL-SENSITIVE — see rescue protocol note below
**Non-diagnostic cases:** fails when structural_lr = 0 or replay_gain = 0

### Rescue protocol (canonical, locked)
- Targeted rescue:         B1_CUE × 3 reps/round, consolidation × 3 passes/round, × 3 rounds
- Generic plain reference: consolidation × 9 passes, no pre-cueing

### SIG-E protocol history and the E017/E018 vs E019 discrepancy
| Experiment | Protocol variant | SIG_E value | Explanation |
|---|---|---|---|
| E017 | Post-cons + post-damage probe cues before rescue | +33 | Probes pre-load E_b; generic rescue benefits |
| E018 | Same as E017 | +33 | Identical |
| E019 | No inter-phase probe cues | +155 | Generic rescue starts cold; near-zero recovery |
| E019R | No inter-phase probe cues (E019 protocol) | see CSV | Same as E019 |

E019 / E019R is the canonical forward protocol. E017/E018 values are preserved as record.
