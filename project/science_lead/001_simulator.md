# e017 — Traceable Simulator Core for Article v2

## Purpose

Rebuild the simulator-facing foundation of the new article so that the branch-resolved model becomes visually inspectable, reproducible, and reviewer-readable.

This experiment does **not** attempt to add new DANDI evidence, new biological claims, or new NDT/MNPS extensions. Its only purpose is to make the existing simulator auditable at the level of time-resolved traces, protected signatures, and theory-to-code mapping.

The motivating reviewer concern is that the previous manuscript described the simulator and comparator results mostly through prose and summary tables, while lacking clear simulation traces or experiment figures. e017 should fix that first.

---

## Core research question

Can the canonical branch-accessibility simulator emit a complete, reproducible trace package showing how fast access, eligibility, consolidation support, slow structural accessibility, recall support, and memory linking evolve across encoding, consolidation, perturbation, and rescue?

---

## Required scope

Implement or refactor only what is needed to expose and save the simulator’s internal dynamics.

Do **not** yet add:

* new DANDI analyses,
* new biological interpretations,
* new manuscript claims,
* large parameter sweeps,
* scaling tests,
* qMRI/NDT/MNPS connections,
* or new open-data bridge sections.

This is an instrumentation and traceability experiment.

---

## Starting point

Use the current canonical simulator setup as the reference condition:

```text
branches: b0, b1, b2, b3
overlap branch: b1
trace mu_1: b0 strong, b1 shared, b2 weak, b3 weak
trace mu_2: b0 weak, b1 shared, b2 strong, b3 weak
encoding: mu_1 cue passes, then mu_2 cue passes
consolidation: replay-dependent slow structural update
primary model: full_model
```

Use existing experiment scripts as reference material, especially:

```text
experiments/exp001_minimal_branch_linking.py
experiments/exp014_structural_gate_ablation.py
experiments/exp015_comparator_baselines.py
```

If the existing implementation differs from this description, do not silently rewrite history. Instead, document the actual current implementation and adapt e017 around the real code.

---

## Primary implementation task

Add a trace-export layer to the simulator.

For each branch and each time step / phase step, export at minimum:

```text
time_or_step
phase
branch_id
trace_id_or_probe_id when relevant
x_b                         # fast branch activity / integration state
fast_access                 # A_f
slow_access                 # A_s or structural-access-derived access
effective_access            # A_b
eligibility                 # E_b
translation_readiness       # P_b or equivalent capture/resource variable
structural_accessibility    # M_b
input_drive                 # I_b if available
replay_drive                # r_b or replay contribution if available
context_value               # if context gating is active
```

For each trace/probe and each relevant step, export:

```text
time_or_step
phase
trace_id
recall_support              # R_mu or implementation-equivalent support
readout_value               # if thresholded/nonlinear readout exists
context_label               # correct, wrong, none, ambiguous if applicable
```

For memory linking, export:

```text
time_or_step
phase
trace_pair
linking_score               # L_mu_nu
overlap_branch_contribution
nonoverlap_contribution
```

The exported data should be saved in machine-readable form, preferably:

```text
results/e017_traceable_simulator_core/traces/branch_traces.csv
results/e017_traceable_simulator_core/traces/trace_support.csv
results/e017_traceable_simulator_core/traces/linking_trace.csv
results/e017_traceable_simulator_core/summary/signature_summary.csv
results/e017_traceable_simulator_core/summary/run_metadata.json
```

CSV is acceptable for first pass. Parquet is optional.

---

## Required phases to trace

At minimum, the canonical run should label these phases:

```text
init
encode_mu_1
encode_mu_2
pre_consolidation_probe
consolidation_replay
post_consolidation_probe
overlap_damage
post_damage_probe
targeted_rescue
post_rescue_probe
```

If current scripts do not yet implement all phases in one run, implement the smallest clean wrapper that reproduces them without changing the underlying mechanism.

---

## Protected signatures to compute

e017 should compute the five protected signatures from the trace outputs, even if the formula is initially simple.

### SIG-A — overlap-branch structural writing

Does the overlap branch gain more slow structural accessibility than non-overlap branches?

```text
delta_M_overlap = M_b1_post_consolidation - M_b1_pre_consolidation
delta_M_nonoverlap_mean = mean(delta_M_b0, delta_M_b2, delta_M_b3)
SIG_A_score = delta_M_overlap - delta_M_nonoverlap_mean
```

### SIG-B — linking gain after consolidation

Does memory linking increase after replay-dependent consolidation?

```text
SIG_B_score = L_mu1_mu2_post_consolidation - L_mu1_mu2_pre_consolidation
```

### SIG-C — context separation

Does correct-context recall exceed wrong-context or no-context recall?

```text
SIG_C_score = R_correct_context - R_wrong_context
```

If context probes are not present in the canonical wrapper, document that SIG-C is pending rather than fabricating a result.

### SIG-D — linking-vs-recall dissociation under overlap damage

Does focal overlap damage hurt linking more than single-trace recall?

```text
linking_drop = L_post_consolidation - L_post_damage
recall_drop = R_single_trace_post_consolidation - R_single_trace_post_damage
SIG_D_score = linking_drop - recall_drop
```

### SIG-E — targeted rescue selectivity

Does targeted overlap rescue improve linking more than generic or non-targeted rescue?

```text
SIG_E_score = L_targeted_rescue - L_generic_or_non_targeted_rescue
```

If generic rescue does not yet exist, create a minimal placeholder comparator only if it can be implemented cleanly. Otherwise mark SIG-E as partially implemented and explain what is missing.

---

## Required figures

Generate first-pass figures directly from exported traces.

Use simple matplotlib figures. Do not focus on visual polish yet.

Required figure files:

```text
results/e017_traceable_simulator_core/figures/Fig_e017_01_branch_state_traces.png
results/e017_traceable_simulator_core/figures/Fig_e017_02_structural_accessibility_traces.png
results/e017_traceable_simulator_core/figures/Fig_e017_03_recall_and_linking_traces.png
results/e017_traceable_simulator_core/figures/Fig_e017_04_signature_barplot.png
```

Figure 1 should show branch-level fast variables, such as `x_b`, `fast_access`, and `effective_access`.

Figure 2 should show slow variables, especially `M_b`, `E_b`, and `P_b` or equivalent.

Figure 3 should show `R_mu1`, `R_mu2`, and `L_mu1_mu2` across phases.

Figure 4 should show SIG-A to SIG-E as a compact first-pass signature summary.

If a variable is not currently available, do not invent it. Add an explicit missing-variable note in the QC report.

---

## Tests and validation

Add or update tests so that e017 is not only a figure script.

Minimum tests:

```text
test_trace_export_has_required_columns
test_trace_export_has_all_expected_phases
test_overlap_branch_identity_is_preserved
test_linking_score_recomputes_from_branch_traces
test_signature_summary_matches_trace_files
test_deterministic_canonical_run_reproduces_same_hash
```

Optional but recommended:

```text
test_no_nan_in_required_trace_columns
test_branch_ids_are_unique_and_complete
test_trace_ids_are_unique_and_complete
test_phase_order_is_monotonic
```

---

## Reproducibility requirements

Save:

```text
config used
git commit hash if available
python version
package versions if easy
random seed
parameter set
branch allocations
trace allocations
phase schedule
output file hashes
```

Write these to:

```text
results/e017_traceable_simulator_core/summary/run_metadata.json
```

Also create:

```text
results/e017_traceable_simulator_core/README.md
results/e017_traceable_simulator_core/qc_report.md
results/e017_traceable_simulator_core/effect_summary.md
results/e017_traceable_simulator_core/claim_ledger.md
results/e017_traceable_simulator_core/figure_manifest.md
```

---

## Claim discipline

e017 may support only these claim types:

### Allowed if successful

```text
The simulator now emits reproducible branch-level and trace-level dynamics.
The canonical run can be inspected as time-resolved traces rather than only summary metrics.
The protected signatures can be computed from saved trace files.
The theory-to-code bridge is more auditable.
```

### Not allowed from e017 alone

```text
The biological model is validated.
The cytoskeletal mechanism is proven.
DANDI evidence supports the simulator.
The model is robust across parameters.
The simulator scales beyond the canonical four-branch case.
The model outperforms all plausible alternatives.
```

Those require later experiments.

---

## Expected deliverables

At the end of e017, report back with:

1. What code was changed.
2. Which script runs the canonical trace export.
3. Where outputs are saved.
4. Whether all required trace variables are available.
5. Which protected signatures are fully computable.
6. Which protected signatures are missing or partial.
7. Whether deterministic rerun produces identical outputs.
8. A compact table:

```text
Signature | Computed? | Direction | Evidence file | Limitation | Next action
```

9. The four figure paths.
10. The next recommended experiment.

---

## Recommended next experiment after e017

If e017 succeeds, the next experiment should be:

```text
e018 — Comparator Trace Matrix
```

Purpose:

Run the same trace-export and SIG-A to SIG-E computation across:

```text
full_model
fast_context_only
replay_no_structure
random_slow_drift
fixed_allocation_only
```

Then produce a comparator matrix showing which mechanisms reproduce the full joint signature profile.

Do not start e018 until e017 produces stable, inspectable canonical traces.
