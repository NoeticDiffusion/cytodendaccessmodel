# Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking

## A simulator-first test of replay-dependent structural writing, specificity, and rescue

Robin Langell
Independent Researcher
For correspondence: [hello@noeticdiffusion.com](mailto:hello@noeticdiffusion.com)

---

## Abstract

Dendritic branches are active computational compartments whose local state can shape whether synaptic input is amplified, stabilized, or later recruited. However, it remains unclear whether memory linking requires only fast contextual gating and synaptic strengthening, or whether recent activity can also leave a slower branch-level accessibility bias that constrains future reuse. Here we introduce a simulator-first framework for testing slow branch-level accessibility as a structural constraint on memory linking. The model separates fast branch access from a slower structural variable, (M_b), updated by eligibility, replay, and consolidation support. In a canonical two-trace motif, replay-dependent slow writing produces overlap-branch strengthening, linking gain, perturbation-sensitive loss, and targeted rescue. These signatures are traceable in exported simulator dynamics and are not reproduced by fast-gating-only, replay-without-structure, random-drift, fixed-allocation, weight-only, global-gain, shuffled-replay, eligibility-only, or resource-only comparators. Robustness analyses show that the joint profile occupies a bounded write–replay regime rather than a single tuned point. Scaling and motif tests show that the mechanism generalizes beyond the four-branch motif, while weak overlap and hub-like universal overlap define expected boundary conditions. These results support slow branch-level accessibility as an executable, falsifiable model-discrimination hypothesis. They do not establish a unique molecular or cytoskeletal memory code.

---

## Author Summary

Memories are often modeled as changes in synaptic weights or recurrent network states. This paper explores a complementary possibility: that dendritic branches may also carry slow access biases that affect which local routes remain available for later stabilization, linking, and recall. We do not claim that dendrites or cytoskeletal structures directly store memory content. Instead, we ask whether a slow branch-level accessibility variable improves a minimal simulator of memory linking.

The revised article is simulator-first. We begin with a canonical branch-resolved model, export its internal traces, and test whether replay-dependent slow structural writing is needed to reproduce a joint profile of overlap-branch strengthening, linking gain, selective perturbation vulnerability, and targeted rescue. We then compare the full model against several simpler alternatives, stress the parameter regime, scale the model beyond four branches, and test motifs with weak, strong, chain, hub, and sparse overlap. The result is bounded: the model works in regimes with sufficient but not universal overlap, fails when overlap is too weak, and loses specificity in hub-like motifs. The value of the framework is therefore not proof of a cytoskeletal memory code, but a clearer and more falsifiable simulator for testing slow dendritic-access constraints.

---

# Introduction

Memory models often emphasize synaptic weights, recurrent population dynamics, and contextual gating. These mechanisms remain central. Yet dendrites and spines are not passive recipients of synaptic input. They regulate local integration, nonlinear amplification, compartmentalization, and plasticity. This raises a narrower question: beyond fast gating and synaptic strengthening, can recent activity leave a slower branch-level accessibility bias that changes which dendritic routes remain easier to stabilize, relink, or recruit later?

The present paper addresses that question through a simulator-first strategy. The aim is not to prove a molecular memory code, nor to identify a unique cytoskeletal mechanism. Instead, the aim is to make the slow-accessibility hypothesis executable and discriminable. The core proposal is that branches carry two kinds of access-relevant state. Fast access determines momentary opening through cue input, local dendritic state, spine state, and contextual gating. Slow structural accessibility, denoted (M_b), provides a persistent branch-level bias that can be modified by local eligibility, replay, and consolidation support.

Associative-memory linking is used here as a deliberately narrow testbed. The broader hypothesis concerns branch-level dendritic accessibility as a general access-and-stabilization constraint, potentially relevant to hippocampal and prefrontal systems. However, memory linking provides an unusually clear operational surface: two or more traces can share branches, replay can preferentially recruit overlap, focal perturbation can target shared routes, and rescue can ask whether restoring overlap-branch access selectively restores linking.

The previous version of this work combined biological framing, simulator results, and open-data bridge analyses in the main article. The present version reorganizes the paper around the simulator. Open-data analyses are moved to the supplementary material and treated as exploratory observable bridges, not as validation of the latent variable (M_b). This revised structure makes the paper’s main claim narrower: a replay-dependent slow branch-level structural variable is required to reproduce a specific joint simulator profile under the tested conditions, whereas simpler comparators reproduce only partial or non-specific effects.

---

# Operational Definitions

**Branch accessibility** refers to how readily a dendritic branch can participate in encoding, consolidation, linking, or recall.

**Fast access** refers to momentary opening driven by cue input, local dendritic/spine state, and context.

**Slow structural accessibility** refers to a persistent branch-level bias variable, (M_b), that changes how easily a branch can be recruited later.

**Slow structural writing** refers to replay-, eligibility-, and consolidation-dependent updating of (M_b).

**Memory linking** refers to increased cross-trace facilitation between memory traces. In the simulator, linking is operationalized as shared structural accessibility between trace allocations.

**Single-trace recall** refers to support for retrieving one trace without requiring cross-trace facilitation.

**Comparator baseline** refers to a model variant that removes, replaces, or scrambles one mechanism to test whether the full model’s joint profile is specific.

**Targeted rescue** refers to post-perturbation restoration of overlap-branch eligibility or structural access, compared against generic or non-targeted rescue.

**Observable bridge** refers to an indirect empirical analysis that tests a downstream consequence of the model without directly measuring (M_b).

---

# Graphical Abstract

**Figure 1. Slow branch-level accessibility as a structural constraint on memory linking.**
Fast dendritic access opens branches moment by moment, while replay-dependent slow structural writing changes which branches remain easier to recruit later. Two traces that share an overlap branch can become linked when replay consolidates that shared branch. Perturbing the overlap branch selectively harms linking, and targeted rescue restores linking only when the slow structural-writing pathway is available. The model tests branch-level accessibility, not direct molecular memory storage.

---

# Claim Map and Evidence Ladder

| Prediction family              | What the slow-accessibility account adds                                                        | Executable diagnostic                              | Main result                                                             | Claim boundary                         |
| ------------------------------ | ----------------------------------------------------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------- |
| Replay-linked selective reuse  | Recent activity should bias later reuse of overlapping branches beyond fixed allocation         | Overlap-branch (M_b) gain and linking increase     | Full model shows overlap writing and linking gain                       | Simulator result, not biological proof |
| Fast-vs-slow separation        | Context separation may arise from fast gating, while slow writing is needed for durable linking | SIG-C versus SIG-A/SIG-B/SIG-E                     | SIG-C is broad and non-diagnostic; SIG-A/B/E are slow-writing dependent | Joint profile matters                  |
| Perturbation-sensitive linking | Linking should be more sensitive to overlap damage than single-trace recall                     | Focal overlap damage and rescue                    | Linking drops after overlap damage; targeted rescue restores            | SIG-D is not diagnostic alone          |
| Comparator discrimination      | Simpler mechanisms should fail the full joint profile                                           | Baseline and hard comparator matrix                | No tested comparator reproduces the full structural-rescue profile      | Alternatives not globally ruled out    |
| Robustness                     | Result should not depend on one tuned parameter point                                           | OAT sweeps and 2D heatmaps                         | Bounded write–replay regime identified                                  | Not arbitrary parameter robustness     |
| Motif generalization           | Mechanism should survive larger allocation spaces but have topology limits                      | 4–32 branches; weak/strong/chain/hub/sparse motifs | Generalizes except weak overlap; hub over-links                         | Not realistic dendritic-tree scaling   |

---

# Conceptual Framework

## Fast access and slow structural bias

The model separates fast and slow access. At the fast timescale, dendritic branches respond to cue input, contextual state, and local branch/spine conditions. At the slower timescale, branch-specific structural accessibility biases whether a branch remains easier to stabilize and reuse after replay or consolidation.

This distinction is important because fast contextual gating can explain some disambiguation effects without requiring slow structural writing. Indeed, the simulator results below confirm that context separation is not diagnostic of slow structural accessibility. The claim is narrower: replay-dependent slow writing is needed for the full joint profile of overlap-branch structural gain, durable linking, and targeted rescue.

## Minimal formalization

The effective accessibility of branch (b) is written as:

[
A_b(t)=A_b^f(t)A_b^s(t)
]

where (A_b^f(t)) is fast access and (A_b^s(t)) is slow access derived from (M_b(t)).

Fast branch activity follows:

[
\tau_x \dot{x}_b=-x_b + A_b(t)[I_b(t)+R_b(t)]
]

where (I_b(t)) is cue input and (R_b(t)) is replay or recurrent recruitment.

Eligibility evolves as:

[
\tau_E \dot{E}_b=-E_b+\phi(x_b,I_b)
]

Branch-local consolidation support evolves as:

[
\tau_P \dot{P}*b=-P_b+\rho*{replay}r_b(t)+\rho_\nu\nu(t)
]

Slow structural accessibility is updated by:

[
\dot{M}*b=\eta E_b(t)P_b(t)W(t)(M*{max}-M_b)-\lambda_M M_b+\epsilon_b(t)
]

Trace-level recall support is:

[
R_\mu(t)=\sum_b a_{\mu b}A_b(t)x_b(t)
]

Memory linking is:

[
L_{\mu\nu}(t)=\sum_b a_{\mu b}a_{\nu b}M_b(t)
]

These equations are phenomenological. (M_b) may summarize several biological processes, including spine geometry, actin remodeling, dynamic microtubule entry, transport readiness, local translation, or metabolic support. It should not be read as a direct molecule count or as proof of a cytoskeletal code.

---

# Results

## 1. A traceable simulator makes the slow-writing hypothesis executable

We first instrumented the canonical branch-resolved simulator so that branch-level and trace-level dynamics could be inspected directly. The canonical model contains four branches, two traces, and one overlap branch shared by both traces. Encoding generates branch-specific eligibility, replay generates consolidation support, and slow structural writing updates (M_b).

The trace export records fast activity (x_b), fast access, slow access, effective access, eligibility (E_b), consolidation support (P_b), structural accessibility (M_b), recall support (R_\mu), and linking (L_{\mu\nu}) across initialization, encoding, pre-consolidation probe, replay consolidation, post-consolidation probe, overlap damage, post-damage probe, targeted rescue, and post-rescue probe.

In the canonical run, the overlap branch showed selective slow structural gain during replay consolidation. The linking score increased after consolidation, decreased after overlap damage, and recovered after targeted rescue. The simulator therefore no longer depends on summary tables alone; the internal dynamics are directly inspectable.

**Article-facing claim:** the simulator emits reproducible branch-level and trace-level dynamics, making the slow-writing hypothesis executable.

**Boundary:** this is an internal simulator validation, not biological validation.

---

## 2. Replay-dependent slow writing produces a joint canonical signature profile

The canonical model was evaluated using five protected signatures:

* SIG-A: overlap-branch structural writing,
* SIG-B: linking gain after consolidation,
* SIG-C: context separation,
* SIG-D: linking-vs-recall dissociation under overlap damage,
* SIG-E: targeted rescue selectivity.

The full model passed all five signatures under the locked canonical protocol. Overlap-branch structural writing was positive, linking increased after consolidation, context separation was present, linking was more vulnerable than single-trace recall under overlap damage, and targeted rescue produced strong normalized recovery relative to generic no-precue consolidation.

However, not all signatures were equally diagnostic. Subsequent analyses showed that SIG-C is largely architectural and context/allocation-related, while SIG-D can arise from overlap-branch perturbation geometry. The diagnostic claim therefore rests on the joint profile, especially the combination of slow structural writing, linking gain, and rescue selectivity.

**Article-facing claim:** replay-dependent slow writing produces a traceable joint profile of overlap strengthening, linking gain, perturbation vulnerability, and rescue.

**Boundary:** SIG-C and SIG-D are not specific markers of slow structural writing when considered alone.

---

## 3. Simpler baseline comparators fail the joint profile

We next compared the full model against baseline comparators:

* fast-context-only,
* replay-without-structure,
* random slow drift,
* fixed allocation only.

All comparators preserved at least some partial behavior. In particular, context separation passed broadly, confirming that SIG-C reflects fast gating or allocation structure rather than slow writing. Several comparators also showed linking-vs-recall dissociation under overlap damage, confirming that SIG-D is not diagnostic alone.

However, no simpler baseline reproduced the full SIG-A to SIG-E profile. Models without structural writing failed overlap-branch (M_b) gain and linking growth. Random slow drift failed specificity. Fixed allocation preserved overlap geometry but failed replay-dependent dynamic updating.

**Article-facing claim:** under canonical parameters, no tested simple baseline reproduced the full joint signature profile.

**Boundary:** this does not rule out all possible alternatives; it tests specified comparators.

---

## 4. Hard comparators reproduce isolated behavioral effects but not the full structural-rescue profile

To make the comparator test fairer, we added harder alternatives:

* Hebbian weight-only,
* soma/global-gain-only,
* shuffled replay,
* eligibility-only,
* resource-only.

These comparators were evaluated at both structural and behavioral-output levels. This distinction is important because a weight-only comparator should not be dismissed merely because it lacks (M_b). The relevant question is whether it reproduces the same behavioral profile and the same structural-accessibility/rescue profile.

Hebbian weight-only produced strong behavioral linking but no branch-specific structural writing and no rescue of a written structural state. Eligibility-only and resource-only comparators produced transient effects but no persistent recoverable structure. Soma/global gain produced broad activation without branch-specific structural writing. Shuffled replay produced a small apparent overlap advantage in the smallest four-branch motif but failed in larger allocation spaces.

Across canonical, strong-overlap, chain-overlap, and sparse-random motifs, no hard comparator reproduced the full structural-accessibility and rescue profile of the full model.

**Article-facing claim:** alternative mechanisms can reproduce isolated behavioral effects, but the tested hard comparators fail the combined structural, behavioral, perturbation, rescue, and specificity profile.

**Boundary:** weight-only and gain-based mechanisms remain serious alternatives for some behavioral outputs, but not for the full structural-accessibility profile as defined here.

---

## 5. One-at-a-time robustness identifies bounded functional ranges

We next varied key parameters one at a time while preserving the locked signature definitions. The model survived meaningful variation in structural learning rate, replay gain, eligibility decay, structural decay, structural noise, context gain, timing gap, overlap strength, and readout threshold.

The result was not uniform success. Failure boundaries were informative. Structural learning rate near zero failed slow writing and linking. Replay gain near zero failed replay-dependent consolidation. Very high structural decay eroded persistence. Long timing gaps combined with eligibility decay impaired consolidation. Weak overlap below approximately 0.40 failed linking. Moderate structural noise was tolerated, but high noise combined with low replay produced instability.

Context gain had little effect on the full profile because SIG-C was already supported by allocation geometry. This reinforced the interpretation that context separation is an auxiliary fast-access signature rather than a diagnostic slow-writing signature.

**Article-facing claim:** the full model’s joint profile is not a single-run artifact; it survives one-at-a-time parameter variation within bounded regimes.

**Boundary:** one-at-a-time robustness does not imply robustness to all parameter interactions.

---

## 6. Two-parameter heatmaps reveal a write–replay regime and temporal eligibility boundary

We then tested biologically meaningful parameter interactions. Six parameter pairs were evaluated:

* structural learning rate × replay gain,
* eligibility decay × timing gap,
* overlap strength × replay gain,
* structural decay × structural learning rate,
* structural noise × replay gain,
* context gain × structural learning rate.

The structural learning rate × replay gain heatmap showed a coherent write–replay regime. Both parameters were required: low structural learning rate or low replay gain caused failure. The eligibility decay × timing gap heatmap showed a temporal boundary: long delays combined with fast eligibility decay exhausted the tag before consolidation. The overlap strength × replay gain heatmap showed that linking requires sufficient overlap and sufficient replay. Structural decay imposed a persistence limit even when learning rate was high. Structural noise was broadly tolerated except at extreme noise and low replay. Context gain again confirmed that context separation is separable from slow structural writing.

**Article-facing claim:** the mechanism occupies a bounded functional regime rather than a single tuned point.

**Boundary:** the tested heatmaps cover specified pairs, not arbitrary high-dimensional parameter combinations.

---

## 7. Scaling and motif tests show generalization with topology-dependent specificity limits

The canonical four-branch motif is useful for exposition, but it could be dismissed as a hand-built toy model. We therefore generalized the simulator to 4, 8, 16, and 32 branches and tested multiple motif classes:

* canonical,
* weak overlap,
* strong overlap,
* chain overlap,
* hub overlap,
* sparse random allocation.

The mechanism generalized beyond four branches in canonical and strong-overlap motifs. Chain-overlap motifs showed local linking stronger than distant linking, with modest leakage. Sparse-random motifs also passed under tested densities and seeds, although density could create universal linking at larger scales.

Weak-overlap motifs failed as expected, confirming the overlap threshold identified in robustness tests. Hub-overlap motifs produced strong structural writing and linking but failed specificity: because all traces shared the same hub branch, all trace pairs became linked. This is not a clean success. It is an over-linking boundary condition.

**Article-facing claim:** the mechanism generalizes beyond the canonical motif in selected non-hub topologies, but specificity depends on overlap structure.

**Boundary:** this is motif generalization, not realistic dendritic-tree scaling.

---

## 8. Shuffled replay fails as branch allocation space expands

The hard-comparator analysis showed a small apparent structural signal for shuffled replay in the smallest four-branch motifs. We therefore audited shuffled replay across branch counts and seeds.

With 20 seeds, the apparent effect was shown to be a high-end small-sample draw rather than a stable signal. In the four-branch motif, shuffled replay reached only about 7% of the full-model overlap advantage on average. At 8 branches the ratio fell below 5%, and by 16–32 branches it approached zero. Variance also decreased with increasing branch count.

This result clarifies the role of replay identity. Replay alone is not sufficient. Identity-preserving replay is required for scalable structural specificity.

**Article-facing claim:** shuffled replay can weakly mimic overlap writing in the smallest allocation space by chance, but this effect decays rapidly with branch count.

**Boundary:** the result concerns the implemented shuffled-replay comparator, not every possible replay-disruption model.

---

# Discussion

## What the simulator supports

The revised simulator supports a bounded, executable claim: replay-dependent slow branch-level structural writing can reproduce a joint profile of memory-linking signatures that tested alternatives do not reproduce. The full model produces traceable overlap-branch strengthening, replay-linked linking gain, perturbation-sensitive loss, and targeted rescue. The result survives one-at-a-time and two-parameter robustness tests, generalizes beyond the four-branch motif, and shows meaningful topology-dependent boundaries.

The strongest contribution is not any single signature. Context separation is not diagnostic of slow writing. Perturbation-sensitive linking is not diagnostic alone. The contribution is the joint profile: structural writing, linking gain, perturbation response, rescue selectivity, robustness, and specificity across non-hub motifs.

## What the simulator does not support

The model does not prove that dendrites store memory content directly. It does not establish a cytoskeletal memory code. It does not identify (M_b) with a specific molecule, organelle, cytoskeletal state, or imaging proxy. It does not demonstrate biological validity in hippocampus or prefrontal cortex. It is a simulator-first model-discrimination framework.

The open-data analyses from the previous manuscript are therefore moved to supplementary material as exploratory observable bridges. They may help identify downstream empirical consequences, but they do not validate (M_b) directly.

## Biological interpretation

The biological interpretation remains plural and cautious. (M_b) may represent a family of slow branch-local constraints: spine geometry, actin remodeling, microtubule invasion, local translation, transport readiness, mitochondrial or metabolic support, or other structural and biochemical processes. The key idea is not that one molecule stores memory, but that branch-local structural state can bias future access.

This framing fits a broader view of dendrites as active, history-sensitive access structures. Hippocampal memory linking provides the current testbed, but similar fast/slow access separations may be relevant in prefrontal contextual control, working memory stabilization, and systems-level consolidation. Those extensions remain speculative until separately implemented and tested.

## Failure modes are informative

Several failures strengthen the model’s interpretability. Weak overlap fails. Excessive hub overlap produces over-linking. Long timing gaps fail when eligibility decays. High structural decay erodes persistence. Shuffled replay fails as allocation space expands. These failures show that the model is not a free-floating explanation that succeeds everywhere. It defines boundary conditions.

## Future tests

The most direct future biological tests would require longitudinal branch-resolved recordings across linked memory formation, replay, perturbation, and rescue. Stronger evidence would come from experiments that dissociate fast contextual gating from slower replay-dependent stabilization; perturb overlap-relevant dendritic mechanisms; and test whether linking is more vulnerable than single-trace recall under targeted structural disruption.

Future simulator work should extend beyond branch-count scaling toward more biologically structured dendritic trees, including proximal/distal organization, apical and basal compartments, spine distributions, inhibitory gating, branch-specific thresholds, and hippocampus–prefrontal differences.

---

# Materials and Methods

## Overview

The revised study is organized as a simulator-first evidence ladder. The primary experiments test whether replay-dependent slow branch-level structural writing is needed to reproduce a joint signature profile in memory linking. Open-data bridge analyses are not part of the main evidence ladder and are described in S3 Appendix.

## Simulator architecture

The canonical simulator contains four branches and two traces. One branch is shared between the traces and serves as the overlap branch. Each trace also has private branches. Cue input drives trace-specific branch activation, eligibility marks recently active branches, replay generates consolidation support, and slow structural writing updates (M_b). Recall support and linking are computed from trace-to-branch allocations and branch accessibility states.

## Trace export

The simulator exports branch-level traces, trace-support traces, and linking traces. Exported branch variables include phase, branch ID, fast activity, fast access, slow access, effective access, eligibility, consolidation support, structural accessibility, input drive, and replay drive. Trace-level outputs include recall support. Pair-level outputs include linking scores and branch contributions.

## Protected signatures

The canonical signatures are:

* SIG-A: overlap-branch structural writing,
* SIG-B: linking gain after consolidation,
* SIG-C: context separation,
* SIG-D: linking-vs-recall dissociation under overlap damage,
* SIG-E: targeted rescue selectivity.

SIG-E is reported as normalized recovery difference, not as percentage points.

## Comparator models

Baseline comparators included fast-context-only, replay-without-structure, random slow drift, and fixed allocation only. Hard comparators included Hebbian weight-only, soma/global-gain-only, shuffled replay, eligibility-only, and resource-only variants. Comparators were evaluated with both structural signatures and behavioral-output metrics so that alternatives lacking (M_b) were not unfairly dismissed.

## Robustness analyses

One-at-a-time sweeps varied structural learning rate, replay gain, eligibility decay, structural decay, structural noise, context gain, timing gap, overlap strength, and readout threshold. Two-parameter heatmaps tested interactions among structural learning rate, replay gain, eligibility decay, timing gap, overlap strength, structural decay, structural noise, and context gain.

## Motif and scaling analyses

A generalized motif engine generated canonical, weak-overlap, strong-overlap, chain-overlap, hub-overlap, and sparse-random allocation motifs. Branch counts ranged from 4 to 32. Multi-trace motifs included chain, hub, and sparse-random structures. Generalized signatures measured expected-overlap structural writing, expected-pair linking gain, perturbation specificity, rescue selectivity, and false-linking or specificity where defined.

## Shuffled replay audit

Shuffled replay was audited across branch counts and seeds to determine whether small-motif apparent structural matching was stable. Full-model reference values were compared against shuffled-replay means, standard deviations, and ratios across 4, 8, 16, and 32 branches.

## Open-data bridge analyses

Open-data analyses using DANDI datasets are reported only in S3 Appendix. They are treated as exploratory observable bridges. They do not directly measure (M_b), dendritic branch accessibility, cytoskeletal writing, or a molecular memory code.

---

# Data and Code Availability

All author-generated code, analysis scripts, configuration files, figures, and manuscript-source material should be made available in the public `cytodendaccessmodel` repository and archived in a manuscript-matched Zenodo snapshot before submission. Empirical datasets used in supplementary bridge analyses should remain referenced through their original public archive records and should not be redistributed unless permitted by their licenses.

---

# Acknowledgments

The author gratefully acknowledges the investigators who generated and shared the public datasets discussed in the supplementary open-data bridge analyses.

The author also acknowledges editorial, analytical, and software-organization assistance from large language models during literature synthesis, structured critique, drafting, and revision. These tools are not authors and did not provide peer review. The author reviewed and approved all scientific claims, code, analyses, citations, and final wording, and takes full responsibility for the manuscript.

---

# Bibliography

[To be renumbered from the previous manuscript.]

Recommended reference groups:

1. Dendritic computation and nonlinear dendrites
2. Spine geometry and compartmentalization
3. Synaptic tagging and capture
4. Local translation and branch-level plasticity
5. Replay and memory consolidation
6. Engram and memory-linking literature
7. Cytoskeletal, actin, microtubule, transport, and metabolic support literature
8. Contextual gating and routing models
9. Associative memory and comparator theory
10. Methods and open-data references for S3 Appendix

---

# Supporting Information Captions

## S1 Appendix. Mathematical framework and theory-to-executable mapping

Extended formalization of the branch-accessibility model, including fast/slow access factorization, eligibility dynamics, consolidation support, slow structural writing, recall support, linking metrics, and the mapping between biological notation and simulator variables. This appendix also documents the interpretation of (M_b) as a phenomenological slow accessibility variable rather than a direct molecular memory substrate.

## S2 Appendix. Simulator architecture, robustness, comparators, and motif tests

Detailed simulator architecture, canonical parameter set, trace-export schema, protected signature definitions, comparator baseline definitions, hard comparator implementations, one-at-a-time robustness, two-parameter heatmaps, motif generator, scaling analyses, specificity metrics, and shuffled replay audit. This appendix contains reproducibility details for E017–E022R.

## S3 Appendix. Exploratory open-data bridge analyses

Exploratory DANDI bridge analyses moved from the main article. These analyses test downstream observable consequences of the branch-accessibility hypothesis, including offline reactivation, structured inter-plane coupling, and perturbation-sensitive cross-day stabilization. They do not directly observe (M_b), dendritic branch accessibility, cytoskeletal memory writing, or molecular memory storage. Their role is to suggest empirical contact points and constraints for future work, not to validate the simulator.
