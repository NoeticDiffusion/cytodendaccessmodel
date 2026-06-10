// A Branch-Resolved Cytoskeletal-Dendritic Accessibility Model of Associative Memory
// Technical Paper
// Author: Robin Langell, 2026

#import "template_eLife.typ": elife-template

#show: doc => elife-template(
  title: [A branch-resolved cytoskeletal-dendritic accessibility model of associative memory],
  authors: [Robin Langell],
  affiliations: [
    #par(first-line-indent: 0pt)[Independent Researcher]
  ],
  correspondence: [`hello@noeticdiffusion.com`],
  abstract: [
    Associative memory is often described in terms of synaptic weights and recurrent population dynamics, yet current biology also suggests that memory expression depends on which dendritic routes remain locally available for stabilization, linking, and later recruitment. We develop a branch-resolved structural-accessibility framework in which fast dendritic and spine states determine momentary access, whereas slower structural variables bias which branches remain easiest to reuse across time. The claim is intentionally narrow: slow structural accessibility is proposed as a falsifiable and executable hypothesis, not as a uniquely established molecular write mechanism.

    We test that hypothesis in two steps. First, we formalize the framework and implement a minimal branch-resolved simulator, asking which signatures require replay-dependent slow writing rather than generic routing or fast contextual gating alone. Second, we use three open datasets as observable bridges rather than as direct microscopic assays. In the simulator, replay-dependent consolidation strengthens overlap branches, memory linking is more fragile than single-trace recall under structural perturbation, and simpler comparator baselines fail to reproduce the joint signature profile. In open data, DANDI `000718` shows modest excess offline enrichment of NeutralExposure-defined core units above a strong population-burst baseline, DANDI `000336` shows reproducibly structured above-null inter-plane coupling with the clearest bilateral access-constraint match in the supplementary cross-area pair, and DANDI `001710` shows lower subject-level cross-day stability in SparseKO than in Cre under a subject-level permutation null, with weaker separation from Ctrl. Taken together, the biology, executable diagnostics, and open-data bridges support slow branch-based structural accessibility as a coherent and empirically contactable research program while leaving its molecular implementation open.
  ],
  doc,
)

= Author Summary

This paper asks whether associative memory depends only on synaptic weights and fast circuit states, or also on slower changes in which dendritic branches remain available for future use. We propose that fast dendritic and spine events control moment-to-moment access, whereas slower structural states bias which branches are easiest to stabilize, link, and recruit over longer timescales. We then test that idea in a deliberately staged way. A minimal branch-resolved simulator asks which signatures genuinely require slow structural writing, and three open neural datasets ask whether downstream observable consequences of that account survive in real recordings. The resulting picture is mixed but informative: the framework is biologically plausible, computationally explicit, and partly supported by open data, yet it does not amount to direct observation of a cytoskeletal memory code. Its value is therefore not closure, but a sharper and more falsifiable bridge between dendritic physiology, consolidation, memory linking, and selective vulnerability to perturbation.


= Introduction

Associative memory must solve a selective-access problem: how local synaptic events become available for encoding, stabilization, linking, and later retrieval in a way that remains context sensitive rather than globally indiscriminate. Classical accounts emphasize synaptic strength and recurrent population dynamics, and recent theory shows that contextual gating of neurons and synapses can substantially improve capacity and specificity @Podlaski2025HighCapacity. At the same time, contemporary neuroscience points to a more layered substrate in which dendrites, spines, replay, local molecular resources, and slower structural constraints shape what can be stored and reactivated @Frankland2005RecentRemoteMemories @Guskjolen2023EngramNeurons @Knierim2016Tracking @Zaki2025Engram @GrellaDonaldson2024LC.

The gap addressed here is narrower than a general appeal to dendrites. Fast contextual gating, synaptic tagging and capture, and dendritic allocation already explain important forms of route selection and trace overlap. What remains less explicit is whether recent activity can leave a branch-specific bias that outlasts the original fast gating or tagging window and thereby changes which dendritic routes remain easiest to stabilize, relink, or recruit later. The central claim of this manuscript is that such a slow structural-accessibility layer is biologically plausible, computationally explicit, and testable, even if its molecular implementation remains open.

The paper proceeds along one evidence ladder. First, it assembles the relevant biology into a branch-resolved framework in which fast dendritic and spine states determine momentary access while slower structural variables provide a learned bias over future recruitability. Second, it asks which signatures of that framework survive in a minimal executable model when confronted with comparator baselines and perturbations. Third, it asks whether three open neural datasets contain downstream observable bridges that are directionally consistent with the same hypothesis. The paper's contribution is therefore not a claim that a cytoskeletal code has already been demonstrated, but a claim that slow branch-based accessibility can be stated clearly enough to be stressed by simulation and by open data.

Dendrites are now understood as active computational compartments rather than passive cables @London2005DendriticComputation @Spruston2008PyramidalNeurons @Sjostrom2008DendriticExcitability. Local NMDA plateaus, $"Ca"^(2+)$ spikes, and branch-specific nonlinearities can amplify clustered input, while spine-neck geometry regulates electrical and chemical compartmentalization @Larkum2009Synaptic @schiller2000nmda @polsky2004computational @Major2013ActiveDendrites @Tonnesen2014 @Araya2014. These findings motivate a shift in emphasis: the biologically relevant access state of memory expression is plausibly a dendritic or spine-level condition, not merely a synaptic weight in isolation.

Several neighboring literatures make a slower bias layer plausible. Synaptic Tagging and Capture (STC), local translation, branch-level plasticity, and dendritic allocation all provide mechanisms by which recent activity can mark particular subcellular compartments as eligible for later stabilization @RedondoMorris2011STC @Ibrahim2024STC @Rogerson2014SynapticTaggingAllocation @Hacisuleyman2024DendriticTranslation @Das2023LocalTranslationMemory @Daskin2025LocalProteinSynthesisSynapses @Govindarajan2011DendriticBranch @Kastellakis2023DendriticEngram. Engram and memory-linking work further suggests that temporally adjacent memories can preferentially reuse overlapping dendritic segments rather than only overlapping cell ensembles @Uytiepo2025EngramArchitecture @Sehgal2025ContextLinking @Choucry2024MemoryLinkingIdentity.

Candidate implementations of a slower bias need not be microtubule-exclusive. Dynamic microtubules can invade active spines, regulate local transport and morphology, and interact with actin-rich spine remodeling @kapitein2011 @hu2008activity @Merriam2011Dynamic @Merriam2013 @Dent2017MicrotubulesMemory @Dent2020DynamicMicrotubulesSynapse @Elie2015TauCoOrganizes. Transport, mitochondrial positioning, and local metabolic support can also remain uneven across dendritic branches @Faits2016 @ChangReynolds2006MitoTrafficking @Misgeld2007DendriticMito @Rangaraju2019MitoCompartments @Thomas2023PostsynapticMito @Bapat2024VAP. The most conservative reading is therefore plural: a slow structural-accessibility variable may summarize several interacting mechanisms rather than a single privileged molecule.

This manuscript consequently treats cytoskeletal and related structural biology as candidate implementation space rather than as settled proof. Fast contextual or circuit-level gating still matters @WangYang2018Routing @Muller2012InhibitoryControlDendriticExcitation @Basu2016LongRangeInhibitionMemory @Tzilivaki2023HippocampalInterneuronsMemory. STC and dendritic allocation still explain short-timescale overlap. The added claim is that a more persistent branch-specific bias should sharpen predictions about replay sensitivity, delayed linking, selective fragility under structural perturbation, and cross-session stabilization.


#align(center)[
  #image("figures/Cytoskeletal dendritic accessibility and memory.png", width: 95%)
]

#align(center)[
  #emph[Graphical Abstract. Overview of the branch-resolved cytoskeletal-dendritic accessibility model of associative memory. Fast dendritic and spine states instantiate graded local access conditions through which synaptic input is amplified, isolated, or propagated, whereas slower structural states provide a dynamic structural accessibility bias over which dendritic routes remain persistently easier to stabilize and later recruit. Local tags, replay-associated consolidation, and branch-specific allocation progressively write a structural accessibility field that shapes encoding, contextual retrieval, and memory linking.]
]

#v(2em)

== Claim Map and Evidence Ladder

The manuscript is organized around three prediction families that are carried forward consistently from theory to simulation to open-data confrontation.

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, left, left, left, left),
  table.header(
    [*Prediction family*], [*What the slow-accessibility account adds*], [*Executable diagnostic*], [*Observable bridge*], [*Main empirical readout*],
  ),
  [Replay-linked selective reuse], [recent activity should bias later reuse of overlapping branches beyond generic routing], [replay-dependent overlap-branch strengthening and linking growth], [offline reactivation of recent ensemble-core units], [DANDI `000718`: modest excess enrichment above a strong burst baseline],
  [Structured rather than indiscriminate access], [branch-level accessibility should support coupling that is organized but not globally open], [branch-specific gain structure and nontrivial route selectivity], [cross-plane coupling above null yet bounded by within-plane coupling], [DANDI `000336`: above-null inter-plane coupling, strongest bilateral match in the supplementary cross-area pair],
  [Perturbation-sensitive long-timescale stabilization], [slow structural writing should be more important for long-timescale linking and stabilization than for immediate recall alone], [linking degrades more strongly than single-trace recall under structural perturbation], [cross-day stabilization deficit under a candidate write-related perturbation], [DANDI `001710`: SparseKO below Cre under the subject-level permutation null, with weaker separation from Ctrl],
)

#par(first-line-indent: 0pt)[
  #emph[Table 1. Claim map used throughout the manuscript. The same prediction families organize the conceptual framework, executable diagnostics, and open-data readouts.]
]


= Conceptual Framework

== Fast access states and slow structural bias

The framework developed here separates two timescales. At the fast timescale, dendritic branches and spines act as semi-autonomous access-control units: local nonlinearities, spine-neck geometry, and immediate circuit context determine whether input is amplified, isolated, or propagated @London2005DendriticComputation @Spruston2008PyramidalNeurons @Major2013ActiveDendrites @Tonnesen2014 @Araya2014. At the slower timescale, structural variables bias which branches remain easiest to reopen, stabilize, and recruit across later events. In this manuscript, `structural accessibility` is the formal label for that slower bias layer.

The mechanistic intuition is deliberately conservative. Structural accessibility need not carry mnemonic content directly. Instead, it can bias fast access through local geometry, transport readiness, and metabolic support. Spine-coupled remodeling changes effective electrical and biochemical coupling to the parent branch, while branch-level transport and organelle positioning change how readily costly plasticity can be sustained @Merriam2013 @Faits2016 @Rangaraju2019MitoCompartments @Thomas2023PostsynapticMito. In this sense, structural accessibility acts as a rheostat beneath faster dendritic events rather than as a symbolic storage register.

This mechanistic contrast is illustrated in @fig-structural-impedance-gating.

#figure(
  image("figures/figure_1_structural_impedance_gating_v3.png", width: 90%),
  caption: [Structural impedance gating at the spine level. In a low-access structural state, reduced support is associated with thinner spine geometry and higher effective impedance, reducing propagation from the synapse to the parent dendrite. In a high-access structural state, activity-dependent structural support enlarges the spine compartment, lowers effective impedance, and facilitates current flow into the dendritic branch. The figure illustrates the paper's core mechanistic claim that slow structural state can bias fast synaptic-to-dendritic access without itself carrying mnemonic content.]
) <fig-structural-impedance-gating>

How could such a bias be written? The working answer adopted here is a replay-compatible consolidation cycle. Recent local activity generates branch-specific eligibility, later instructional or modulatory conditions enable consolidation, and repeated replay or reactivation preferentially stabilizes branches that were both recently active and consolidation-eligible. This logic is close in spirit to STC-like and neoHebbian three-factor learning rules, but is expressed here at branch level rather than only at the level of isolated synapses @fremaux2016neuromodulated @RedondoMorris2011STC @Gerstner2018EligibilityTraces.

The proposed write logic is summarized in @fig-cytoskeletal-learning-cycle.

#figure(
  image("figures/figure_2_Cytoskeletal learning cycle_v2.png", width: 85%),
  caption: [The cytoskeletal learning cycle. Fast dendritic and synaptic activity generates local tags and calcium-dependent eligibility signals at specific branches and spines. During later consolidation windows, including replay- and sleep-associated periods, these transient signals are converted into slower structural updates of the accessibility field. In this view, structural remodeling does not replace synaptic plasticity, but contributes a slower write process through which branch-level accessibility becomes progressively stabilized.]
) <fig-cytoskeletal-learning-cycle>

Candidate biological implementations include actin-dominated spine remodeling, dynamic microtubule invasion, Tau- and MAP-linked stabilization, and branch-local transport or metabolic support @Hotulainen2010ActinSpines @Borovac2018ActinDynamicsSpines @Dent2017MicrotubulesMemory @Dent2020DynamicMicrotubulesSynapse @Elie2015TauCoOrganizes. Sleep-associated consolidation windows are a plausible setting in which local tags could be converted into more durable branch-level biases, because spindle-rich sleep couples dendritic calcium activity, local translation, and structural remodeling @Ulrich2016SleepSpindles @PeyracheSeibt2020Spindles @Seibt2017DendriticSpindles @Niethard2021SleepCalcium @Yang2014SleepSpines @Sun2020SleepPlasticity @Wang2024SleepDependentEngramReactivation. These mechanisms motivate the framework, but the paper does not require any single one of them to be uniquely correct.

== Minimal biological formalization

The formal goal is not a full biophysical reconstruction, but a restrained language for the hypothesis. We denote fast branch integration by $x_b(t)$, local spine accessibility by $s_i(t)$, slow branch structure by $M_b(t)$, branch eligibility by $E_b(t)$, and trace-level recall support by $R_mu(t)$. The key separation is that fast accessibility depends jointly on immediate dynamics and on a slower structural term:

$ A_b(t) = A_b^f(x_b(t), s(t), C(t)) A_b^s(M_b(t)) $

where $C(t)$ denotes optional fast contextual or circuit-level gating. The factorization is schematic rather than literal, but it makes the intended claim explicit: fast local and circuit dynamics determine momentary opening, while the slow structural state changes how readily that opening can be expressed.

We then write branch dynamics as

$ dot(x)_b = A_b(t) F_b(x, I, s) $

with synaptic drive $I$ and local spine dynamics summarized by

$ dot(s)_i = G_i(s, x, M) $.

At the slower timescale, branch accessibility is updated by bounded, tag-dependent consolidation:

$ dot(M)_b(t) = eta E_b(t) sigma(delta(t) - theta_delta) (1 - M_b(t) / M_"max") - lambda_M M_b(t) + sqrt(2 T_("eff")) xi_b(t) $

where $E_b$ marks recently active branches, $delta(t)$ is a delayed instructional or modulatory signal, the bounded term captures finite local consolidation capacity, and the decay and noise terms capture ongoing turnover in an active structural substrate. This is a phenomenological write rule rather than a literal biochemical rate law.
In that phenomenological spirit, $theta_delta$ should be read as a coarse-grained write-enable threshold summarizing saturation-like biochemical conditions for consolidation, such as CaMKII-, capture-, or neuromodulator-dependent gating @Yagishita2014DopamineTiming. Likewise, $M_"max"$ is not intended as a literal molecule count, but as a compact way to represent finite local structural capacity: only so much branch-specific support can be stabilized before turnover, transport limits, and active-matter-like nonequilibrium constraints begin to dominate @fodor2016 @needleman2017.

The formal relationships among fast access, slow structural bias, and retrieval support are summarized in @fig-minimal-biological-formalization.

#figure(
  image("figures/figure_3_minimal_biological_formalization_v3.png", width: 90%),
  caption: [Minimal biological formalization of fast access and slow structural bias. Fast branch integration states ($x_b$), local spine-access states ($s_i$), and effective branch accessibility ($A_b$) define the moment-to-moment opening of dendritic routes, whereas the slow structural state ($M_b$) and local eligibility traces ($E_b$) determine which branches remain persistently easier to recruit and consolidate. Retrieval support for a trace ($mu$) depends on whether the branches on which that trace depends are both active and accessible at recall.]
) <fig-minimal-biological-formalization>

Retrieval support is modeled as

$ R_mu(t) = sum_b a_(mu b) A_b(t) x_b(t) $

where $a_(mu b)$ describes how strongly trace $mu$ depends on branch $b$. A trace can therefore fail not because its synapses are absent, but because the required branch subset is insufficiently accessible when recall is attempted. The branch-allocation intuition behind linking and contextual retrieval is illustrated in @fig-memory-linking-contextual-retrieval.

#figure(
  image("figures/figure_4_memory linking and contextual retrieval_v2.png", width: 90%),
  caption: [Branch allocation as a mechanism for memory linking and contextual retrieval. Memories formed close in time can preferentially recruit overlapping dendritic segments that remain in relatively high structural accessibility states. Under this view, later recall of one trace can partially facilitate recruitment of another, not only through cellular ensemble overlap but also through shared branch-level allocation. More independent memories rely on less-overlapping branch subsets and therefore remain more separable at retrieval.]
) <fig-memory-linking-contextual-retrieval>

Four modeling choices keep the framework bounded:

1.  #strong[Accessibility, not explicit content coding:] the model concerns which branches and spines are available to participate in encoding and recall, not a claim that structural variables alone encode mnemonic content.
2.  #strong[Phenomenological branch variables:] $x$, $s$, $M$, and $R$ compress many biophysical processes into coarse-grained states rather than directly measured single-molecule quantities.
3.  #strong[Fast and slow terms are schematic:] the factorized access term is a compact way of stating the hypothesis, not a claim that all mechanisms are cleanly separable in vivo.
4.  #strong[Molecular implementation remains open:] cytoskeletal, transport, metabolic, and translation-linked mechanisms are treated as candidate contributors to the slow bias layer.

= Results

== Executable stress tests of slow structural accessibility

The conceptual framework above becomes informative only if it can be instantiated without collapsing into generic routing, arbitrary recurrence, or unconstrained symbolic linkage. The executable model used here is therefore deliberately minimal: a four-branch system with one overlap branch shared between two traces, fast access dynamics, local eligibility traces, replay-dependent slow updates, and comparator baselines that remove or replace slow structural writing. The simulator is not intended as a biophysical reconstruction of a full neuron; it is a mechanistic testbed for asking which signatures are diagnostic of a slow structural-accessibility layer and which follow more generically from architecture alone.

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, left, left),
  table.header(
    [*Signature family*], [*Representative result*], [*Interpretive status*],
  ),
  [Branch-specific structural writing], [$Delta M_("b1") = +0.21758$ on the overlap branch and a $43.1%$ rise in the linking metric], [diagnostic when compared against comparator baselines and structural ablation],
  [Context-sensitive recall], [correct-context support `0.5051` vs `0.2982`; no-context retrieval shows recency bias], [real but partly attributable to the fast contextual layer],
  [Three-factor consolidation dependence], [spacing weakens branch writing, selective replay underconsolidates the non-replayed trace, and zero modulatory drive collapses linking to `0.40142`], [supports the replay-plus-write-enable logic],
  [Vulnerability and rescue profile], [linking degrades more strongly than recall under focal overlap damage and targeted rescue loses selectivity when structural writing is removed], [one of the strongest diagnostic signatures of the slow layer],
  [Robustness and scope], [`100%` directional pass rates on five protected claims; only the full model passes the joint comparator panel; hub-overlap ablation reduces linking by `89%`], [shows the mechanism is not a single favorable run],
)

#par(first-line-indent: 0pt)[
  #emph[Table 2. Condensed executable signature profile. The most informative results are the replay-dependent strengthening of overlap branches, the greater fragility of linking than of single-trace recall under structural damage, and the failure of simpler comparators to match the full joint profile.]
]

These executable results matter unequally. Context-sensitive disambiguation is partly architecturally expected once a fast contextual term is included. By contrast, overlap-branch strengthening, linking growth after consolidation, and targeted rescue selectivity become informative precisely because they weaken or disappear when slow structural writing is removed. The simulator therefore does not prove the biology, but it does show that the proposed branch-resolved hypothesis survives stronger mechanistic stress tests than several simpler alternatives. Complementary one-at-a-time robustness sweeps over `structural_lr`, `replay_gain`, `eligibility_decay`, `structural_noise`, timing gap, and contextual bias preserve the directional sign of the protected metrics across the tested ranges, arguing against a knife-edge parameterization. Full parameter tables, baseline definitions, and reproducibility details are provided in `S2 Appendix`.

== Open-data design and analysis registry

Open datasets do not directly reveal a latent slow structural field. The relevant empirical question is narrower: do the theory and executable model leave downstream signature families that can be measured with fixed, inspectable pipelines? The open-data program in this paper therefore serves as an observable bridge from a latent branch-resolved hypothesis to concrete neural measurements rather than as direct microscopic validation.

The three datasets were selected because they probe different points on the evidence ladder. DANDI `000718` addresses offline ensemble reactivation and memory-linking-related reuse across sessions. DANDI `000336` provides near-simultaneous somatic and distal dendritic recordings suitable for testing structured inter-plane coupling @Amaya2026DANDI000336. DANDI `001710` couples longitudinal virtual-reality behavior to a perturbation of postsynaptic membrane fusion, making it useful for asking whether a candidate write-related disruption selectively weakens long-timescale stabilization @Plitt2026DANDI001710. Compact robustness and QC summaries for all three dataset families are collected in `S3 Appendix`.

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, left, left, left, left),
  table.header(
    [*Dataset*], [*Reviewable question*], [*Primary endpoint*], [*Primary null or control*], [*Retained claim boundary*],
  ),
  [DANDI `000718`], [Do offline high-synchrony events preferentially reactivate recent ensemble-core units?], [event-versus-inter-event core-unit enrichment], [duration-matched inter-event windows plus registration-shuffle controls], [positive but modest excess above a strong population-burst background],
  [DANDI `000336`], [Is inter-plane coupling structured rather than indiscriminate?], [cross-plane coupling `r` and `z` versus circular-shift null], [within-window circular-shift nulls and within-plane comparisons], [above-null inter-plane coupling across all analyzed pairs, with the cleanest bilateral access-constraint match in the supplementary cross-area pair],
  [DANDI `001710`], [Does a candidate write-related perturbation selectively reduce long-timescale stabilization?], [subject-level cross-day similarity by genotype], [subject-level permutation null plus lag and channel robustness], [SparseKO lower than Cre, weaker separation from Ctrl, and quantitatively channel-sensitive rather than channel-broken],
)

#par(first-line-indent: 0pt)[
  #emph[Table 3. Analysis registry for the open-data component. Each dataset is introduced by a single reviewable question, a primary endpoint, a primary null or control, and a bounded interpretive claim.]
]

Because the open-data program spans multiple datasets and related sub-analyses, the reported p-values are used as analysis-specific anchors rather than as a familywise-confirmatory survey across the entire paper. Claims are therefore bounded within each pipeline, while threshold sweeps, channel comparisons, and related checks are treated as robustness analyses rather than as separate confirmatory endpoints. The null constructions are empirical and dataset specific: in `000718`, each detected event is compared against ten duration-matched inter-event windows from the same offline session and then stressed against registration-shuffle controls; in `000336`, one plane is circularly shifted within each window `200` times before window-level nulls are aggregated into condition-level `z` scores; in `001710`, genotype contrasts are evaluated against `1000` subject-level label permutations that preserve group sizes.

== Observable bridge A: offline replay-linked reactivation in DANDI `000718`

The `000718` analysis asks whether offline population events preferentially reactivate units central to a recent experience. We analyzed three NeutralExposure-to-offline session pairs using cross-session ROI registration, ensemble extraction, and event-versus-inter-event enrichment scoring @Zaki2024OfflineLinking @Sheintuch2017CellRegistration @Vergara2025CaliAli @Molter2018DetectingAssemblies @Nagayama2022NMFAssemblies @Shen2022Deconvolution. NeutralExposure ensembles were extracted with non-negative matrix factorization (`k = 8`), core units were defined as the top `15%` by weight, and each detected high-synchrony offline event was compared against ten duration-matched inter-event windows from the same offline session @NavasOlive2024RipplAI @Liu2022ECannula.

Across all three tested session pairs, NeutralExposure-defined core units showed positive event-versus-inter-event enrichment (`+0.0535` to `+0.0609`, `z = 10.9` to `20.9`) with consistent direction across activity-threshold sweeps. Registration-shuffle controls retained most of the generic event-related activation, showing that the absolute effect sits on top of a strong population-burst baseline. The retained interpretation is therefore deliberately narrow: `000718` supports modest excess enrichment above that burst baseline for NeutralExposure-defined core units during high-synchrony offline events. It does not by itself establish sequence-level replay or direct observation of a slow structural write process.
The retained enrichment boundary is summarized in @fig-open-data-000718-enrichment.

#figure(
  image("figures/figure_6_open_data_000718_enrichment.png", width: 95%),
  caption: [Open-data evaluation of DANDI `000718`. NeutralExposure-defined core units show positive event-versus-inter-event enrichment across all three tested session pairs, while registration-shuffle controls retain most of the generic population-burst baseline. The figure makes the paper's intended claim boundary explicit: the surviving signal is consistent but modest, and is best interpreted as excess enrichment above a strong event-related background rather than as direct replay-sequence proof.]
) <fig-open-data-000718-enrichment>

Threshold robustness across the tested activity cutoffs is summarized in @fig-open-data-000718-threshold-sweep.

#figure(
  image("figures/figure_7_open_data_000718_threshold_sweep.png", width: 92%),
  caption: [Threshold robustness for the DANDI `000718` enrichment result. Across all three NeutralExposure-to-offline session pairs, event-versus-inter-event enrichment remains positive over the tested activity thresholds (`0.0`, `0.5`, `1.0 sigma`). The figure therefore supports the narrower methodological claim that the retained enrichment signal is not an artifact of a single activity cutoff, even though its magnitude remains modest relative to the broader population-burst baseline.]
) <fig-open-data-000718-threshold-sweep>

== Observable bridge B: structured inter-plane coupling in DANDI `000336`

The `000336` analysis asks whether communication between paired imaging planes is structured rather than indiscriminate. Same-session coupling analyses were extended to all six NWB files, organized as three within-session pairs @Amaya2026DANDI000336. The retained pipeline used exact timestamp alignment when planes shared acquisition timing, `0.5 s` binning for the supplementary interleaved cross-area session, within-window circular-shift nulls (`n = 200`), and block-level merging of short stimulus conditions. Seven condition families were common to the full bundle, although usable windows varied by pair because the pipeline retained a conservative minimum block-duration rule.

The main result is pair-level and bounded. Cross-plane coupling remained above the circular-shift null in all three spontaneous comparisons and across all tested stimulus conditions, while the stricter `cross < both within` signature was fully met only in the supplementary cross-area pair and remained partial in the two primary cross-depth pairs because one within-plane estimate was unusually weak or unusually strong. The retained claim is therefore not that all pairs cleanly satisfy the strongest bilateral criterion, but that the full analyzed bundle supports reproducibly structured above-null inter-plane coupling, with the strongest clean access-constraint match appearing in the supplementary cross-area pair.

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, left, center, center, center, center, center, left),
  table.header(
    [*Pair*], [*Geometry*], [*ROIs (A/B)*], [*Cross r*], [*z vs null*], [*Within A*], [*Within B*], [*Verdict*],
  ),
  [pair_a], [cross-depth], [6 / 62], [0.0197], [4.98], [0.0169], [0.0377], [partial],
  [pair_b], [cross-depth], [4 / 25], [0.0295], [4.04], [0.1160], [0.0261], [partial],
  [pair_c], [cross-area], [27 / 53], [0.0224], [5.16], [0.0275], [0.0434], [positive],
)

#par(first-line-indent: 0pt)[
  #emph[Table 4. Summary of the DANDI `000336` full-bundle spontaneous coupling result. All three within-session pairs show cross-plane coupling above the circular-shift null. Only the supplementary cross-area pair cleanly satisfies `cross < both within`, whereas the two cross-depth pairs remain partial because of within-plane asymmetry.]
]

Condition-level coupling across the full analyzed bundle is shown in @fig-open-data-000336-coupling-by-condition.

#figure(
  image("figures/figure_8_open_data_000336_coupling_by_condition.png", width: 98%),
  caption: [Open-data evaluation of full-bundle DANDI `000336` by condition. Across all three analyzed bundle pairs, cross-plane coupling remains above the circular-shift null in every plotted condition family. In the two primary cross-depth pairs, the stricter `cross < both within` criterion is only partially met because one within-plane population is unusually weak or unusually strong, whereas the supplementary cross-area pair provides the cleanest bilateral access-constraint match. The figure therefore supports a bounded access-constraint claim across the full available bundle rather than a hand-picked subject subset.]
) <fig-open-data-000336-coupling-by-condition>

== Observable bridge C: perturbation-sensitive cross-day stabilization in DANDI `001710`

The `001710` analysis asks whether perturbing a candidate postsynaptic write-related mechanism selectively weakens cross-day stabilization while sparing broader within-session structure. The analysis was extended from an earlier four-subject bridge to a broader `23`-subject genotype bundle (`7` Cre, `9` Ctrl, `7` SparseKO; `139` NWB files total) @Plitt2026DANDI001710. Subject-level cross-day summaries retained `4` to `6` usable days per subject and yielded group means of `0.3374` for Cre, `0.2926` for Ctrl, and `0.2623` for SparseKO. Under the implemented subject-level permutation null, SparseKO lay below Cre (`obs_diff = -0.0751`, `z = -2.1495`, `p = 0.009`), whereas the comparison against Ctrl was directionally similar but weaker (`obs_diff = -0.0303`, `z = -1.2856`, `p = 0.099`). Lag profiles remained lower for SparseKO across lags `1` to `5`, and the long-lag gap widened at lag `5` (`0.1973` in SparseKO versus `0.3634` in Cre and `0.2785` in Ctrl).

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, center, center, left, left),
  table.header(
    [*Group*], [*Subjects*], [*Mean cross-day similarity*], [*Within-session structure*], [*Interpretive role*],
  ),
  [Cre], [7], [0.337], [positive; high split-half reliability], [highest-stability reference group],
  [Ctrl], [9], [0.293], [positive; broader heterogeneity], [non-KO baseline],
  [SparseKO], [7], [0.262], [positive; quantitatively channel-sensitive], [lowest-stability perturbation group],
)

#par(first-line-indent: 0pt)[
  #emph[Table 5. Summary of the broadened DANDI `001710` genotype bundle. Group means are computed from subject-level off-diagonal similarity summaries, with individual subjects contributing `4` to `6` usable days. SparseKO is lowest on the canonical first-channel pass and falls below Cre under the implemented subject-level permutation null.]
]

A dedicated robustness package also shows that the `001710` result is not completely channel-invariant. Across SparseKO subjects, channel `1` was on average somewhat more cross-day stable than channel `0` (`0.3296` versus `0.2623`) and slightly more arm-separated (`0.3119` versus `0.2796`), while both channels retained high split-half reliability. The main-text use of `ch0` is therefore a predefined one-channel-per-animal bookkeeping rule rather than a claim that `ch0` is uniquely privileged biologically. The retained interpretation remains bounded: `001710` supports a broadened subject-level cross-day stabilization deficit in SparseKO, strongest relative to Cre, but channel sensitivity and partial-session heterogeneity argue against treating the dataset as a closed-form proof of a unique structural-write mechanism.

== Cross-dataset synthesis

The open-data picture is uneven but informative. `000718` provides positive yet modest evidence for replay-linked selective reuse above a strong burst baseline. `000336` provides the cleanest article-level bridge for structured rather than indiscriminate access, although its strongest bilateral criterion is only fully met in the supplementary cross-area pair. `001710` provides the strongest long-timescale perturbation bridge, because the cross-day stabilization deficit extends across subjects and separates SparseKO from Cre under the implemented null, even though the comparison against Ctrl is weaker and the result is channel-sensitive. Taken together, the three datasets do not identify a latent $M_b$ field directly, but they do bound which observable consequences of the slow-accessibility account currently survive contact with real recordings.

A compact full-bundle summary of the spontaneous `000336` result across all analyzed pairs is provided in @fig-open-data-000336-summary, making the pair-level consistency visible without implying a separate replication cohort.

#figure(
  image("figures/figure_9_open_data_000336_replication.png", width: 88%),
  caption: [Full-bundle summary for the spontaneous DANDI `000336` signature. All three analyzed bundle pairs show positive cross-plane coupling above the circular-shift null. Only the supplementary cross-area pair cleanly satisfies `cross < both within`, whereas the cross-depth pairs remain partial because of within-plane asymmetry. The figure therefore emphasizes that the above-null coupling signature survives across the full available bundle without erasing the difference between partial and full access-constraint evidence.]
) <fig-open-data-000336-summary>

= Discussion

== What the joint profile supports

The manuscript's central claim is intentionally restricted: slow branch-based structural accessibility is a coherent, executable, and empirically contactable hypothesis for associative memory, not a uniquely demonstrated biological mechanism. In this framing, dendritic branches and spines provide fast access states, whereas structural, transport, metabolic, and related variables provide a slower background bias on which local pathways remain easiest to recruit over time.

What makes that claim useful is the joint profile rather than any single result. The biological literature supports the needed ingredients for such a framework, including active dendrites, spine-level compartmentalization, branch-local allocation, local tagging and translation, and replay-linked consolidation @London2005DendriticComputation @Spruston2008PyramidalNeurons @Larkum2009Synaptic @Major2013ActiveDendrites @Tonnesen2014 @Rogerson2014SynapticTaggingAllocation @Hacisuleyman2024DendriticTranslation @Kastellakis2023DendriticEngram. The executable model shows that these ingredients can be combined into a branch-resolved mechanism whose strongest signatures are not reproduced by simpler baselines. The open-data analyses then show that several downstream bridges survive reproducible measurement, even though they remain partial and indirect. The value of the framework therefore lies less in any single headline result than in the fact that the same slow-accessibility variable gives one mechanistic explanation for replay-linked reuse, structured inter-compartment coupling, and selective long-timescale fragility under perturbation without fitting each dataset separately.

== Alternatives, limitations, and discriminating next tests

Several simpler explanations remain viable for parts of the observed profile. Soma- or synaptic-weight-centered models could reproduce some macroscopic observables without positing a slow branch-specific write variable. Fast contextual gating can explain part of the disambiguation story already @WangYang2018Routing @Keller2020ContextualModulation @Bos2025GainModulation. Dendritic allocation without persistent structural accessibility could explain short-range overlap and some memory-linking effects. In `001710`, a more general plasticity or stability impairment could also reduce cross-day similarity without uniquely targeting a structural write process.

The limitations are therefore central rather than incidental. None of the open datasets directly measure a slow structural accessibility field. The executable model is phenomenological and compressed relative to biology. The `000718` effect is positive but modest relative to the underlying burst baseline. The `000336` result is cleaner, but still belongs to a narrow dataset family. The `001710` bridge is stronger than in the earlier subject-limited version, yet it remains bounded by weaker separation from Ctrl, indirect arm-label validation, partial day coverage in some subjects, and channel sensitivity within SparseKO. These constraints do not defeat the framework; they define where it remains most vulnerable.

The most discriminating future tests are branch resolved rather than purely behavioral. Stronger support would come from longitudinal tracking of identified dendritic segments across linked memories, perturbations that dissociate fast contextual gating from slower replay-dependent stabilization, and interventions showing that cross-session linking degrades more strongly than within-session single-trace recall under structural disruption. The framework will stand or fall on whether such experiments can distinguish a genuine slow accessibility field from explanations based on fast gating or generic plasticity alone. That is also what makes the present proposal scientifically useful: it converts a broad biological intuition into a falsifiable path forward.

= Materials and Methods

== Overview

The operational methods are organized to match the manuscript's evidence ladder. The executable component asks which signatures require replay-dependent slow structural writing in a minimal branch-resolved simulator. The open-data component asks whether three fixed, inspectable pipelines recover downstream signature families that are directionally consistent with the same framework. Extended parameter tables, QC summaries, and robustness outputs are provided in `S2 Appendix` and `S3 Appendix`.

== Executable simulator

The executable results were produced by a branch-resolved Python simulator implemented in `src/cytodend_keylock/` and exposed through experiment scripts in `experiments/`. The canonical simulator uses four dendritic branches, one overlap branch shared between two traces, fast access dynamics, local eligibility traces, replay-dependent slow structural updates, and a delayed write-enable term. The main executable summary in Table 2 draws on the canonical parameter set together with targeted perturbation and comparator runs, including settings that remove slow structural writing, replace it with simpler routing logic, or ablate overlap-related structure. In the formal notation, recall support is written as $R_mu(t) = sum_b a_(mu b) A_b(t) x_b(t)$ to keep the role of accessibility explicit. In the simulator implementation, the same access factor is absorbed earlier into branch activity, so readout is computed from access-gated $x_b$ rather than by multiplying accessibility twice at the final readout step; this is an implementation-order simplification rather than a change in conceptual role, and is mapped explicitly in `S1 Appendix`.

The intended role of the simulator is mechanistic discrimination rather than biophysical realism. The key executable readouts are overlap-branch structural change, linking strength, context-sensitive recall, perturbation vulnerability, rescue selectivity, and robustness across protected claims. Readers wishing to reproduce or extend the numerical results should begin with `experiments/exp001_minimal_branch_linking.py`, `experiments/exp014_structural_gate_ablation.py`, `experiments/exp015_comparator_baselines.py`, and the robustness scripts summarized in `S2 Appendix`.

== DANDI `000718` pipeline

The `000718` pipeline targets offline replay-linked reuse. Three NeutralExposure-to-offline session pairs were analyzed using cross-session ROI registration, ensemble extraction, and event-versus-inter-event enrichment scoring @Sheintuch2017CellRegistration @Vergara2025CaliAli @Nagayama2022NMFAssemblies @NavasOlive2024RipplAI. NeutralExposure ensembles were extracted with non-negative matrix factorization (`k = 8`), core units were defined as the top `15%` of weights within each ensemble, and offline high-synchrony events were identified from population synchrony. Each detected event was compared against ten duration-matched inter-event windows from the same offline session. At the event level, the retained Preferential Reactivation Index uses a size-matched random-core null (`null_n = 500`), and at the session level the real registration is stress-tested against repeated registration-shuffle controls. Threshold sweeps (`0.0`, `0.5`, `1.0 sigma`) were treated as robustness checks rather than separate confirmatory endpoints.

== DANDI `000336` pipeline

The `000336` pipeline targets structured inter-plane access. All six locally available NWB files were organized as three within-session pairs: two primary cross-depth pairs and one supplementary cross-area pair @Amaya2026DANDI000336. Exact timestamp alignment was used when planes shared acquisition timing. For the supplementary `ses-1245548523` cross-area session, `0.5 s` binning was used because timestamps were interleaved rather than shared. Short stimulus presentations were merged into block-level windows under a conservative minimum-duration rule. Observed cross-plane coupling was compared against within-plane coupling and within-window circular-shift nulls (`n = 200`), generated by rolling one plane against the other by nonzero offsets within each analysis window before aggregating window-local nulls to the condition level. The primary endpoint was above-null cross-plane coupling; the stricter `cross < both within` pattern was treated as a stronger but secondary access-constraint criterion.

== DANDI `001710` pipeline

The `001710` pipeline targets perturbation-sensitive long-timescale stabilization. A broad `23`-subject bundle (`7` Cre, `9` Ctrl, `7` SparseKO; `139` NWB files total) was processed using generalized I/O utilities that support both single-channel and multi-channel sessions. Trials were reconstructed from 2P-aligned behavior channels (`trial start`, `trial end`, `trial number`, `block`, `left or right`) cross-referenced against the embedded `trial_cell_data` annotation blob. Conservative `df` matrices were extracted together with aligned behavior, and occupancy-normalized tuning curves were computed for each ROI separately within each session or channel.

Subject-level observables included within-day left-versus-right arm population-vector structure, cross-day similarity from index-matched ROI tuning curves, genotype-level permutation nulls, and day-lag profiles. The main group null uses `1000` subject-level label permutations without replacement while preserving observed group sizes, and reports one-sided empirical p-values in the observed direction. For SparseKO subjects, the canonical group comparison used `ch0` as a predefined one-channel-per-animal summary so that each animal contributed one observation to the genotype contrast; channel-level robustness was summarized separately. Full QC outputs, lag profiles, arm-label audit notes, and channel comparisons are reported in `S3 Appendix`.

== Statistical analysis and reporting conventions

Analyses were run in Python `>=3.10` using the dependencies declared in `pyproject.toml`, including `numpy`, `pyyaml`, `torch`, `dandi`, `pynwb`, and `matplotlib`. The executable simulator is deterministic in its canonical configuration unless a robustness run explicitly enables noise or drift; where seeds are required, they are stated in `S2 Appendix`.

For empirical null-separation summaries, the nominal alpha level was `0.05`. The reported tests are analysis specific rather than familywise across the entire paper. In `000718`, the main quantities are event-versus-inter-event enrichment deltas and registration-shuffle contrasts across `3` NeutralExposure-to-offline session pairs. In `000336`, the main quantities are cross-plane `r`, within-plane `r`, and `z` versus the within-window circular-shift null across `3` within-session pairs (`6` NWB files total). In `001710`, the main inferential step is a subject-level permutation null on genotype contrasts using `23` subjects drawn from `139` NWB files.

The `001710` genotype contrasts are evaluated directionally for the planned ordering `SparseKO < comparator`; the `000718` and `000336` null-separation summaries are likewise directional tests of positive excess enrichment or positive above-null coupling rather than symmetric two-sided screens for any difference. No global multiple-testing correction was applied across all datasets because the three dataset families address different questions and use different null constructions. Exclusions followed pipeline-specific QC and data-availability rules rather than outcome-driven filtering: short `000336` stimulus fragments were merged or omitted if they failed the minimum block-duration rule, and `001710` subjects contributed `4` to `6` usable days depending on recording coverage. Missing data were handled conservatively by dropping unavailable sessions, windows, or days from the relevant summaries rather than imputing them.

Effect sizes are reported throughout as enrichment deltas, coupling coefficients, null-separated `z` scores, and subject-level mean differences. Because several analyses rely on empirical nulls, small bundles, or both, the manuscript emphasizes these effect sizes together with pipeline-specific null-separation summaries rather than forcing a single asymptotic confidence-interval style across all result families.

= Data and Code Availability

All author-generated code, analysis scripts, configuration files, figures, and manuscript-source material necessary to inspect and reproduce the analyses reported in this study are publicly available in the `cytodendaccessmodel` GitHub repository (`https://github.com/NoeticDiffusion/cytodendaccessmodel`) and in the manuscript-matched archived snapshot at Zenodo (DOI `10.5281/zenodo.20615268`; `https://doi.org/10.5281/zenodo.20615268`). The code is released under the GNU General Public License v3.0 (`GPL-3.0`). The empirical analyses use openly available third-party datasets from the DANDI Archive (`000718`: `https://dandiarchive.org/dandiset/000718`, `000336`: `https://dandiarchive.org/dandiset/000336`, `001710`: `https://dandiarchive.org/dandiset/001710`), which are not redistributed by the author and should be accessed through their original archive records. Derived manuscript-facing outputs are organized under `data/dandi/triage/000718`, `data/dandi/triage/000336`, and `data/dandi/triage/001710`. Primary reproduction entry points are organized under `experiments/`, `configs/`, `src/cytodend_keylock/`, `src/dandi_io/`, and `src/dandi_analysis/`, with dataset-specific QC and robustness artifacts summarized in `S3 Appendix`.

= Acknowledgments

The author gratefully acknowledges editorial and analytical assistance from large language models during literature synthesis, structured critique, drafting, and revision. These tools are not authors and did not provide peer review.

The author also gratefully acknowledges the investigators who generated and openly shared the DANDI datasets used in this study, including Yosif Zaki, Denise J. Cai, Jason E. Pina, Jerome A. Lecoq, Joel Zylberberg, and Mark Plitt, together with their collaborators.

#bibliography("references_cytoskeletal_dendritic_accesibility_model.bib")

#pagebreak()

= Supporting Information Captions

S1 Appendix. Mathematical framework and executable bridge. Extended formalization of the structural-accessibility model, including compact state-space equations, factorized accessibility, optional resource-capture extensions, and the theory-to-executable mapping used to relate the biological formalism to the simulator.

S2 Appendix. Executable simulator architecture, parameters, and comparator baselines. Detailed simulator architecture, canonical parameter set, encoding and consolidation protocol, baseline definitions, and stochastic reproducibility notes for the branch-resolved model.

S3 Appendix. Open-data pipelines and reproducibility details. Dataset-specific pipeline descriptions, robustness summaries, subject-level QC notes, null tests, and repository entry points for DANDI `000718`, `000336`, and `001710`.

Associated supporting-information files: `S1_Appendix.pdf`, `S2_Appendix.pdf`, and `S3_Appendix.pdf`.
