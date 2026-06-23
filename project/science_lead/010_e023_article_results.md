# E023 — Article Figure Assembly and Results Draft

## Purpose

Assemble the simulator-first evidence package for article v2.

E017–E022R established:

```text
E017   traceable canonical simulator dynamics
E018   baseline comparator discrimination
E019   one-at-a-time parameter robustness
E019R  locked signature protocol and SIG-E rescue audit
E020   two-parameter robustness heatmaps
E021   scaling and motif generalization
E021R  specificity-aware motif interpretation
E022   hard comparator models
E022R  shuffled replay scaling audit
```

E023 should not add new science unless a missing figure requires a minor replot.

The task is to build the article-facing figure spine and first Results draft.

---

## Core research question for the article

The article should now answer:

```text
Does a replay-dependent slow branch-level structural-accessibility variable explain
a joint profile of memory-linking signatures better than fast gating, fixed allocation,
random drift, weight-only learning, global gain, shuffled replay, eligibility-only,
or resource-only alternatives?
```

The claim should remain bounded:

```text
The simulator supports slow branch-level structural writing as an executable
model-discrimination hypothesis. It does not prove a cytoskeletal memory code.
```

---

## Required article-facing figures

Create a new directory:

```text
article/v2_figures/
```

or, if article outputs are kept under results:

```text
results/e023_article_figure_assembly/
```

Generate publication-facing versions of the following figures.

---

## Figure 1 — Model concept and executable mapping

Purpose: solve the “conceptual framing” problem.

Panels:

```text
A. Biological sketch: branch-level access, fast access, slow structural accessibility
B. Minimal simulator: branches, traces, overlap branch, replay, damage, rescue
C. Equation-to-code mapping: A_f, A_s, M_b, E_b, P_b, R_mu, L_mu_nu
D. Claim boundary: accessibility model, not direct molecular memory proof
```

This figure should make clear:

```text
fast gating ≠ slow structural writing
trace overlap ≠ memory proof
M_b = latent slow accessibility variable, not direct molecule count
```

---

## Figure 2 — Canonical simulator traces

Source: E017.

Show:

```text
branch activity / fast access
eligibility E_b
resource/capture P_b
structural accessibility M_b
recall support R_mu
linking L_mu_nu
```

Main message:

```text
Replay-dependent consolidation selectively increases overlap-branch accessibility
and linking; damage reduces linking; targeted rescue restores linking.
```

---

## Figure 3 — Comparator discrimination matrix

Source: E018 + E022.

Should include two layers:

```text
A. Structural signature matrix: SIG-A to SIG-E
B. Behavioral-output matrix: linking, context, damage, rescue, specificity
```

Comparators:

```text
full_model
fast_context_only
replay_no_structure
random_slow_drift
fixed_allocation_only
hebbian_weight_only
soma_global_gain_only
shuffled_replay
eligibility_only
resource_only
```

Main message:

```text
Some comparators reproduce isolated behavioral effects, but none reproduce
the full structural-accessibility and rescue profile.
```

Important: do not overstate. Especially note:

```text
SIG-C is architectural/contextual, not diagnostic of slow writing.
SIG-D is perturbation-sensitive but not diagnostic alone.
The joint profile matters.
```

---

## Figure 4 — Robustness landscape

Source: E019 and E020.

Panels:

```text
A. One-at-a-time robustness summary
B. structural_lr × replay_gain heatmap
C. eligibility_decay × timing_gap heatmap
D. overlap_strength × replay_gain heatmap
E. structural_noise × replay_gain heatmap
```

Main message:

```text
The model occupies a bounded functional regime, not a single tuned point.
```

Important boundaries to highlight:

```text
structural_lr < ~0.06 fails writing/linking
replay_gain < ~0.25 fails replay-dependent writing
timing_gap ≥ 16 with high eligibility decay fails consolidation
overlap_strength < ~0.40 fails linking
high structural decay imposes persistence limits
```

---

## Figure 5 — Scaling, motifs, and specificity

Source: E021 and E021R.

Panels:

```text
A. Motif schematics: canonical, strong, weak, chain, hub, sparse
B. Generalized signature matrix
C. False-linking / specificity by motif
D. Branch-count scaling: 4, 8, 16, 32
```

Main message:

```text
The mechanism generalizes beyond the four-branch motif, but only within
specific overlap-topology boundaries.
```

Important language:

```text
weak_overlap = expected failure
hub_overlap = mechanistic pass but specificity failure
chain_overlap = local linking with leakage
sparse_random = density-dependent specificity
```

---

## Figure 6 — Shuffled replay audit

Source: E022R.

Panels:

```text
A. gSIG-A full model vs shuffled replay by branch count
B. shuffled/full ratio by branch count
C. seed spread
```

Main message:

```text
Replay alone is insufficient. Identity-preserving replay is required for scalable
structural specificity.
```

Suggested article sentence:

```text
Shuffled replay weakly mimicked overlap-branch writing in the smallest four-branch
motif because random reassignment can revisit the same branch by chance. This
effect decayed rapidly as branch allocation space increased, approaching zero by
16 branches.
```

---

## Results section draft structure

Create:

```text
article/v2_results_draft.md
```

with this structure:

```text
Results

1. A branch-accessibility simulator makes the slow-writing hypothesis executable
2. Replay-dependent slow writing produces traceable overlap-branch strengthening
3. The joint signature profile separates slow writing from simpler baselines
4. Robustness analyses identify a bounded write–replay regime
5. Motif scaling shows generalization with topology-dependent specificity limits
6. Hard comparators reproduce isolated behavioral effects but not the full structural-rescue profile
7. Shuffled replay fails as branch allocation space expands
```

Each subsection should contain:

```text
Question
Experiment/source
Main result
Interpretive boundary
Article-facing claim
```

---

## Required claim ledger synthesis

Create:

```text
article/v2_claim_ledger.md
```

Use this format:

```text
Claim | Status | Evidence | Limitation | Article wording
```

Required claims to classify:

```text
The simulator emits reproducible traces.
The full model passes canonical SIG-A–E.
The tested baseline comparators fail the joint profile.
SIG-C is architectural/contextual, not diagnostic of slow writing.
SIG-D is perturbation-sensitive but not diagnostic alone.
The joint profile survives OAT parameter variation.
The joint profile survives bounded two-parameter regimes.
The mechanism generalizes beyond four branches.
Weak overlap is a predicted failure.
Hub overlap is an over-linking boundary.
Hard comparators can reproduce isolated behavioral effects.
No tested hard comparator reproduces the full structural-rescue profile.
Shuffled replay fails as branch count increases.
The model validates cytoskeletal memory biology.
```

The last claim should be marked:

```text
Not supported
```

---

## Required abstract stub

Create:

```text
article/v2_abstract_stub.md
```

Draft a restrained abstract of 180–230 words.

It should not mention DANDI.

It should include:

```text
branch-level dendritic accessibility
minimal simulator
slow structural writing
fast gating and comparator baselines
robustness and motif scaling
bounded interpretation
not direct molecular validation
```

---

## Required tests / checks

This is mostly documentation and figure assembly, but still add a simple script/check if possible:

```text
tests/test_e023_article_assembly.py
```

Minimum checks:

```text
test_all_required_figures_exist
test_results_draft_exists
test_claim_ledger_exists
test_abstract_stub_exists
test_claim_ledger_marks_biological_validation_as_not_supported
test_figures_have_manifest_entries
```

---

## Required outputs

Create:

```text
article/v2_figure_manifest.md
article/v2_results_draft.md
article/v2_claim_ledger.md
article/v2_abstract_stub.md
article/v2_next_gaps.md
```

`article/v2_next_gaps.md` should list:

```text
What remains before manuscript rewrite
What remains before DANDI can be reintroduced
What remains before submission
```

---

## Acceptance criteria

E023 is complete when:

1. Six article-facing figures are assembled or first-pass generated.
2. A Results draft exists.
3. A claim ledger exists.
4. An abstract stub exists.
5. All claims are epistemically labeled.
6. The figure spine tells a coherent simulator-first story.
7. No new overclaim is introduced.
8. Tests/checks pass.

---

## Recommended next step after E023

Proceed to:

```text
E024 — Manuscript v2 Skeleton
```

Purpose:

Build the new paper around the simulator-first evidence package.

DANDI should remain paused until the simulator-first article spine is coherent.
