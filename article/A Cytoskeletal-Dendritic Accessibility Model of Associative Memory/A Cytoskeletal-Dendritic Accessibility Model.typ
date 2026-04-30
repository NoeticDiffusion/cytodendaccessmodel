// A Cytoskeletal-Dendritic Accessibility Model of Associative Memory
// Technical Paper
// Author: Robin Langell, 2026

#import "template.typ": essay-template

#show: doc => essay-template(
  title: [A cytoskeletal-dendritic accessibility model of associative memory],
  short_title: [Structural accessibility in associative memory],
  author: "Robin Langell",
  affiliation: "Independent Researcher",
  corresponding_email: "hello@noeticdiffusion.com",
  doc,
)

= Abstract

Associative memory is often framed in terms of synaptic weights and recurrent population dynamics. We develop a narrower, branch-resolved alternative in which memory expression also depends on a slower accessibility structure distributed across dendritic branches and spines. In this framework, fast dendritic and synaptic events instantiate momentary access states, whereas slower structural variables bias which dendritic subunits remain easiest to stabilize, link, and later recruit. The contribution is not a demonstrated molecular mechanism, but a computationally explicit, biologically motivated, and falsifiable framework that can be stress-tested against executable and open-data signatures.

In a minimal branch-resolved simulator, replay-dependent consolidation writes branch-specific structural accessibility, linking is more fragile than single-trace recall under structural perturbation, and simpler comparator baselines fail to reproduce the joint signature profile. In public data, DANDI `000718` shows a modest excess enrichment of NeutralExposure-defined core units above a strong population-burst baseline, DANDI `000336` shows structured above-null inter-plane coupling across all analyzed bundle pairs with the clearest bilateral access-constraint match in the supplementary cross-area pair, and DANDI `001710` shows lower subject-level cross-day stability in SparseKO than in Cre under a subject-level permutation null, with a weaker separation from Ctrl. Taken together, these results support slow branch-based accessibility as a coherent, executable, and empirically contactable hypothesis rather than as a uniquely established biological write mechanism.

= Author Summary

We asked whether associative memory depends only on synaptic weights or whether it also depends on slower changes in how dendritic branches remain available for later use. We develop a model in which fast dendritic and spine events control moment-to-moment access, whereas slower structural states bias which branches are easiest to stabilize, link, and retrieve over time. We then test that idea in two ways. First, we implement a branch-resolved simulator and ask which signatures require slow structural writing rather than generic routing alone. Second, we analyze three open neural datasets for downstream signatures predicted by the framework. We find that the model is biologically plausible, executable, and partly supported by public data, but we do not claim direct observation of a slow cytoskeletal memory code. Instead, we argue that structural accessibility is a useful and falsifiable hypothesis that connects dendritic physiology, consolidation, memory linking, and selective vulnerability to perturbation.


= Introduction

The central problem addressed in this manuscript is biological associative memory: how local synaptic events become selectively accessible for encoding, stabilization, linking, and later retrieval in a way that remains context sensitive rather than globally indiscriminate. Classical memory theory often emphasizes synaptic strength and recurrent attractor dynamics, and recent formal work demonstrates that context-dependent gating of neurons and synapses can substantially improve memory capacity and context-specificity in such networks @Podlaski2025HighCapacity, but contemporary neuroscience points to a more layered substrate in which dendrites, spines, local molecular resources, replay, and slower structural constraints are themselves active determinants of what can be stored and reactivated @Frankland2005RecentRemoteMemories @Guskjolen2023EngramNeurons @Knierim2016Tracking @Zaki2025Engram @GrellaDonaldson2024LC.

This manuscript now treats that problem across three linked but distinct epistemic levels. First, it develops a biological theory: dendritic branches and spines define fast access states, whereas a slower structural accessibility layer acts as a dynamic bias over which local routes remain easier to stabilize, link, and later retrieve. Second, it asks whether that logic can be instantiated in an executable model without collapsing into generic routing or generic attractor storage. Third, it asks which parts of that executable bridge survive first confrontation with open neural datasets. The paper is therefore not a trilogy recap but a single claim ladder from biology, to executable mechanism, to empirical signature testing. The main claim is correspondingly narrow: slow branch-based accessibility is presented here as a coherent, executable, and empirically contactable hypothesis that survives initial stress tests better than several simpler alternatives, not as a uniquely demonstrated mechanism.

Dendrites are now understood as active computational compartments rather than passive cables @London2005DendriticComputation @Spruston2008PyramidalNeurons @Sjostrom2008DendriticExcitability. Local NMDA plateaus, $"Ca"^(2+)$ spikes, and branch-specific nonlinearities can amplify clustered input, while spine-neck geometry regulates electrical and chemical compartmentalization @Larkum2009Synaptic @schiller2000nmda @polsky2004computational @Major2013ActiveDendrites @Tonnesen2014 @Araya2014. Recent work further shows that pathway-specific gating can arise within pyramidal neurons through nonuniform dendritic organization, allowing different afferent streams to be modulated independently @Olah2025HCNGating. These findings motivate a shift in emphasis: the biologically relevant access state of memory expression is plausibly a dendritic or spine-level condition, not merely a synaptic weight in isolation. Mechanistically, it is better understood as a graded local accessibility condition rather than as a rigid yes-no latch.

At the same time, evidence from cytoskeletal biology suggests that these fast access states may themselves be biased by a slower structural substrate. Dynamic microtubules invade active spines, regulate spine enlargement, position signaling molecules, and constrain local transport and metabolic support @kapitein2011 @hu2008activity @Merriam2011Dynamic @Merriam2013 @Lemieux2012CaMKII_MT @Faits2016. More mainstream syntheses of dendritic microtubule dynamics now make it possible to describe this layer in a non-exotic way: microtubules act near the synapse as regulators of morphology, transport, and plasticity, while Tau helps coordinate microtubule-actin cross-talk rather than serving as a purely microtubular stiffener @Dent2017MicrotubulesMemory @Dent2020DynamicMicrotubulesSynapse @Elie2015TauCoOrganizes. The strongest interpretation of these findings is not that microtubules directly encode memory content — as has been proposed in more direct formulations of cytoskeletal memory @Priel2010NeuralCytoskeleton — but that they participate in a branch-level accessibility structure: a slowly changing set of constraints that influences which dendritic subunits are most available for stable plasticity and later reactivation.

This view becomes more compelling when integrated with Synaptic Tagging and Capture (STC), local translation, and CaMKII-centered accounts of long-term plasticity @RedondoMorris2011STC @Ibrahim2024STC @Rogerson2014SynapticTaggingAllocation @Lemieux2012CaMKII_MT @Hacisuleyman2024DendriticTranslation @Das2023LocalTranslationMemory @Daskin2025LocalProteinSynthesisSynapses @Gerstner2018EligibilityTraces. These mechanisms provide a biologically grounded way for recent activity to mark specific branches or spines as eligible for durable change, while leaving other nearby compartments relatively untouched. Critically, protein-synthesis-dependent LTP consolidates preferentially at the branch level rather than at isolated synapses, making the dendritic branch a natural unit for long-term stabilization @Govindarajan2011DendriticBranch. Engram studies and work on memory linking further suggest that such allocation is not only cellular but also subcellular: memories formed close in time can preferentially recruit overlapping dendritic segments, and long-term traces are stabilized through synapse-specific and branch-specific remodeling @Uytiepo2025EngramArchitecture @Sehgal2025ContextLinking @GrellaDonaldson2024LC @Kastellakis2023DendriticEngram @Choucry2024MemoryLinkingIdentity.

The present framework builds directly on these neighboring accounts, but its novelty claim is narrower and more specific than a general appeal to dendrites. STC already explains tagging and capture within a bounded temporal window. Dendritic allocation and clustered synaptic overlap already explain why nearby memories can recruit overlapping subcellular compartments. Fast contextual gating already explains moment-to-moment route selection under changing circuit states. What structural accessibility adds is the hypothesis of a more persistent branch-specific bias that can outlast the original tagging window and thereby sharpen predictions about replay sensitivity, delayed linking, and selective fragility under structural perturbation. That is the manuscript's main novelty claim, not the assertion that an already established molecular mechanism has been identified.

Importantly, this slow accessibility hypothesis should be understood as complementary to faster circuit-level gating rather than as its replacement. Pathway selection can also be imposed by inhibitory and disinhibitory motifs acting on dendrites and hippocampal subfields over shorter timescales @WangYang2018Routing @Muller2012InhibitoryControlDendriticExcitation @Basu2016LongRangeInhibitionMemory @Tzilivaki2023HippocampalInterneuronsMemory. The present proposal is that cytoskeletal-dendritic structure adds a slower, learned bias beneath these faster gates. In the manuscript below, `structural accessibility` is used as the formal label for that slower bias layer, while `dynamic bias` and `slow structural gating bias` are used for its mechanistic interpretation.

This distinction matters because much of the relevant biology is already partly explained without invoking a dedicated slow structural layer. Active dendrites, inhibitory motifs, contextual bias, synaptic tagging and capture, and dendritic allocation can already account for moment-to-moment route selection, branch-local amplification, and some forms of learning-dependent trace overlap. What the present framework adds is a more explicit long-timescale accessibility variable that could bias which branch subsets remain easier to stabilize, relink, or later recruit even after the original fast gating state has passed. The discriminating prediction is therefore not simply that dendrites gate memory, but that a history-dependent structural bias should leave branch-specific signatures in linking, contextual retrieval, replay-dependent consolidation, and vulnerability to perturbation that are not exhausted by fast gating alone.

In that sense, the paper is aimed at the core PLOS Computational Biology intersection: a computationally explicit, biologically motivated, falsifiable, and reproducible framework whose strongest value lies in disciplined claim sharpening rather than in rhetorical breadth.

The aim of the present paper is intentionally restrained. We do not argue that microtubules by themselves compute mnemonic content, nor do we require a consciousness-centered framework for the biological argument to stand. Instead, we synthesize literature on dendritic integration, spine compartmentalization, cytoskeletal plasticity, tagging and capture, engrams, and contextual retrieval into a falsifiable hypothesis about how associative memory may be biologically implemented, then ask whether that hypothesis survives executable and empirical sharpening. Any broader geometric or noetic interpretation should be treated as a later theoretical layer, not as a premise of the present manuscript.

The paper is organized across a background tier plus three working evidence levels:

- *Established background:* active dendrites, spine compartmentalization, local tagging and capture, local translation, and dendritic or branch-level memory allocation.
- *Biologically grounded synthesis:* these findings are jointly consistent with a slower branch-level accessibility structure beneath fast dendritic dynamics.
- *Executable sharpening:* a branch-resolved simulator can preserve this logic and reveal which signatures are architecturally expected versus mechanistically diagnostic.
- *Open-data confrontation:* the slow field is not directly observed, but downstream signature families in `000718`, full-bundle `000336`, and a broader subject-level `001710` perturbation bridge can still confirm, refine, or narrow the surviving theory.


#align(center)[
  #image("figures/Cytoskeletal dendritic accessibility and memory.png", width: 95%)
]

#align(center)[
  #emph[Graphical Abstract. Overview of the cytoskeletal-dendritic accessibility model of associative memory. Fast dendritic and spine states instantiate graded local access conditions through which synaptic input is amplified, isolated, or propagated, whereas slower cytoskeletal and structural states provide a dynamic structural accessibility bias over which dendritic routes remain persistently easier to stabilize and later recruit. Local tags, replay-associated consolidation, and branch-specific allocation progressively write a structural accessibility field that shapes encoding, contextual retrieval, and memory linking.]
]

#v(2em)

== Testable Predictions

To move beyond theory, we retain three falsifiable hypotheses that organize the downstream claims of the paper.

=== Hypothesis 1: Cytoskeletal Perturbation Changes Branch-Level Accessibility
*Prediction:* Pharmacological or genetic perturbation of cytoskeletal organization will alter dendritic gain and branch-specific integration, not merely global excitability.
- *Stabilizers / support enhancers:* Should increase persistence of selected access states, but at high levels may produce over-rigid routing and reduced flexibility.
- *Destabilizers / transport disruptors:* Should reduce the reliability of dendritic amplification, spine stabilization, and branch-specific recruitment.
- *Test:* Combine cytoskeletal perturbation with dendritic electrophysiology or calcium imaging while measuring local NMDA amplification, plateau propensity, or branch-to-soma coupling.

=== Hypothesis 2: Memory Linking Depends on Shared Structural Allocation
*Prediction:* Memories formed close in time should be more strongly linked when they recruit overlapping dendritic segments and compatible structural support states.
*Mechanism:* Local tags, shared resource capture, and repeated reallocation to the same branch should increase the probability that later recall recruits partially overlapping traces. Recent work on dendritic engrams and review-level accounts of memory linking strengthen this interpretation by suggesting that overlap is organized not only at the cell-ensemble level but also at the level of dendritic compartments and synaptic allocation @Sehgal2025ContextLinking @Uytiepo2025EngramArchitecture @Kastellakis2023DendriticEngram @Choucry2024MemoryLinkingIdentity.
*Test:* Track dendritic segments and spine clusters across temporally adjacent memory episodes while perturbing cytoskeletal or translation-dependent consolidation.

=== Hypothesis 3: Structural Accessibility Shapes Contextual Retrieval
*Prediction:* Retrieval and memory differentiation should depend not only on cue quality and recurrent network state, but also on whether the relevant dendritic branches remain structurally accessible.
*Mechanism:* Slow structural bias changes which branch ensembles are most available for recall, thereby influencing whether contextual cues favor stable retrieval, remapping, or interference @Miranda2024CA3Remapping @GrellaDonaldson2024LC.
*Test:* Combine contextual retrieval tasks with branch-resolved imaging, local plasticity perturbations, or cytoskeletal/Tau manipulations to determine whether retrieval quality covaries with branch-specific accessibility markers.

These hypotheses are intended as falsifiable probes of the structural accessibility framework. Positive results would support the view that memory access depends on a slow structural layer beneath fast dendritic and synaptic dynamics; negative or mixed results would be equally informative by clarifying where the model overreaches or where alternative mechanisms dominate.

= Results

== Executable Instantiation and Signature Hierarchy

The biological theory above remains intentionally mechanistic and hypothesis driven, but the argument benefits from an executable counterpart. A central question is whether a structural accessibility account can be rendered executable without collapsing into generic routing, arbitrary recurrence, or unconstrained symbolic linkage. The simulator used here is a deliberately minimal branch-resolved system: a small set of dendritic branches carries fast access dynamics, local eligibility traces, and replay-dependent slow structural updates. It is therefore not a biophysical reconstruction of a full neuron, but a mechanistic testbed for asking which signatures depend on slow structural writing and which follow from broader architecture alone. To test that, the same state variables introduced in the biological formalization were implemented in a branch-resolved simulator in which fast branch and spine dynamics determine momentary opening, while replay-dependent slow structural writing alters later accessibility. The value of that executable step is not performance optimization. It is claim sharpening.

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, left, left),
  table.header(
    [*Signature family*], [*Representative result*], [*Interpretive status*],
  ),
  [Branch-specific structural writing], [$Delta M_("b1") = +0.21758$ on the overlap branch and a $43.1%$ rise in the linking metric], [mechanistically informative when compared against simpler baselines and structural ablation],
  [Context-sensitive recall], [correct-context support `0.5051` vs `0.2982`; no-context retrieval shows recency bias], [real but partly attributable to the fast contextual layer],
  [Three-factor consolidation dependence], [spacing weakens branch writing, selective replay underconsolidates the non-replayed trace, and zero modulatory drive collapses linking to `0.40142`], [supports the replay-plus-write-enable logic],
  [Vulnerability and rescue profile], [linking degrades more strongly than recall under focal overlap damage and targeted rescue loses selectivity when structural writing is removed], [one of the strongest signatures of the slow structural layer],
  [Robustness and scope], [`100%` directional pass rates on five protected claims; only the full model passes the joint comparator panel; hub-overlap ablation reduces linking by `89%`], [shows the mechanism is not a single favorable run],
)

#par(first-line-indent: 0pt)[
  #emph[Table 1. Condensed executable signature profile retained in the unified manuscript.]
]

These results matter unequally. Some outcomes, such as context-sensitive disambiguation, are partly architecturally expected once a fast contextual term is included. Others become informative only under perturbation. Comparator baselines and structural-gate ablations indicate that overlap-branch strengthening, linking growth after consolidation, and targeted rescue selectivity are the most diagnostic signatures of replay-driven slow structural writing. In other words, the executable model does not prove the biology, but it does show that a branch-resolved structural-accessibility hypothesis survives stronger mechanistic stress tests than several simpler alternatives.

Full details of the simulator — including the canonical parameter set, network architecture, encoding and consolidation protocol, and comparator baseline definitions — are summarized in `S2 Appendix`. Readers wishing to reproduce or extend the numerical results should begin with `experiments/exp001_minimal_branch_linking.py` and `experiments/exp015_comparator_baselines.py` in the public `cytodendaccessmodel` repository and the manuscript-matched Zenodo snapshot.

== Open-Data Evaluation

Open datasets do not directly reveal a slow structural accessibility field. The relevant question is therefore narrower: do the biological and executable hypotheses leave downstream signature families that can be measured with fixed, reproducible analysis pipelines? The open-data program in this paper therefore serves as an observable bridge from a latent structural-accessibility theory to concrete neural measurements rather than as direct microscopic validation.

=== Dataset Overview

The three datasets used here were chosen because they probe distinct observable consequences of the structural accessibility account rather than because they directly measure the latent slow field itself. DANDI `000718` is a mouse calcium-imaging dataset on offline ensemble co-reactivation and memory linking across days, making it relevant for the paper's question of replay-linked integration between temporally separated traces. DANDI `000336` is the Allen Institute OpenScope Dendritic Coupling project, with near-simultaneous two-photon measurements of somata and distal apical dendrites during visual stimulation, making it suitable for testing whether inter-plane coupling is structured rather than indiscriminate @Amaya2026DANDI000336.

DANDI `001710` comes from the study *Hippocampal place code plasticity in CA1 requires postsynaptic membrane fusion* and combines longitudinal two-photon imaging with virtual-reality behavior in a perturbation framework centered on CA1 `Stx3` deletion. This makes it useful for asking whether a putative structural-write disruption selectively weakens cross-day stabilization while sparing broader contextual or spatial coding features @Plitt2026DANDI001710. Together, the three datasets span offline linking, structured inter-compartment communication, and longitudinal place-code plasticity, providing a more diverse empirical stress test than any single paradigm could offer.

=== Open-Data Results

Compact robustness and QC summaries for all three dataset families are collected in `S3 Appendix`, so that the main text can keep the claim boundaries readable without suppressing the underlying quality checks.

Because the open-data program spans multiple datasets and several related sub-analyses, the reported p-values should be read as analysis-specific anchors rather than as a familywise-confirmatory survey across the entire paper. We therefore use them to bound claims within each dataset-specific pipeline, not to assert a single globally corrected significance statement across all open-data results.

To ask whether offline population events preferentially reactivate units central to a recent experience, we turned to DANDI `000718`. We analyzed three NeutralExposure-to-offline session pairs using cross-session ROI registration, high-synchrony offline event detection, and ensemble-defined core-unit enrichment @Zaki2024OfflineLinking @Sheintuch2017CellRegistration @Vergara2025CaliAli @Molter2018DetectingAssemblies @Nagayama2022NMFAssemblies @Shen2022Deconvolution. NeutralExposure ensembles were extracted with non-negative matrix factorization (NMF; `k = 8`), core units were defined as the top `15%` by weight, offline events were identified from population synchrony, and each event was compared against ten duration-matched inter-event windows from the same offline session. Across all three tested session pairs, NeutralExposure-defined core units showed positive event-versus-inter-event enrichment (`+0.0535` to `+0.0609`, `z = 10.9` to `20.9`) with consistent direction across activity-threshold sweeps. Registration-shuffle controls retained most of the event-related activation, indicating that the surviving effect is modest above a strong population-burst baseline, but the real mappings still exceeded shuffled controls by approximately `+0.005` to `+0.008` per pair @NavasOlive2024RipplAI @Liu2022ECannula. Biologically, the relevant point is therefore not a large absolute effect size, but that a small positive excess remains detectable even after a deliberately strong burst-matched control. The appropriate claim is restrained: `000718` supports modest excess enrichment above burst baseline for NeutralExposure-defined core units during high-synchrony offline events, not a completed proof of sequence-level replay.
The retained enrichment boundary is summarized in @fig-open-data-000718-enrichment.

#figure(
  image("figures/figure_6_open_data_000718_enrichment.png", width: 95%),
  caption: [Open-data evaluation of DANDI `000718`. NeutralExposure-defined core units show positive event-versus-inter-event enrichment across all three tested session pairs, while registration-shuffle controls retain most of the generic population-burst baseline. The figure makes the paper's intended claim boundary explicit: the surviving signal is consistent but modest, and is best interpreted as excess enrichment above a strong event-related background rather than as direct replay-sequence proof.]
) <fig-open-data-000718-enrichment>

Threshold robustness across the tested activity cutoffs is summarized in @fig-open-data-000718-threshold-sweep.

#figure(
  image("figures/figure_7_open_data_000718_threshold_sweep.png", width: 92%),
  caption: [Threshold robustness for the DANDI `000718` enrichment result. Across all three NeutralExposure-to-offline session pairs, event-versus-inter-event enrichment remains positive over the tested activity thresholds (`0.0`, `0.5`, `1.0 sigma`). The figure therefore supports the narrower methodological claim that the retained H1 signal is not an artifact of a single activity cutoff, even though its magnitude remains modest relative to the broader population-burst baseline.]
) <fig-open-data-000718-threshold-sweep>

To ask whether communication between paired imaging planes is structured rather than indiscriminate, we turned to DANDI `000336`. Same-session coupling analyses were extended to all six NWB files, organized as three within-session pairs @Amaya2026DANDI000336. The retained pipeline used exact timestamp alignment when planes shared acquisition timing, `0.5 s` binning for the supplementary interleaved cross-area session, within-window circular-shift nulls (`n = 200`), and block-level merging of short stimulus conditions. Seven condition families were common to the full bundle, but usable windows varied by pair; in `sub-656228 / ses-1247233186`, the movie protocol split forward and reverse segments too finely to survive the conservative block-merging rule. The main H3 result is therefore best summarized at the pair level: cross-plane coupling remains above the circular-shift null in all three spontaneous comparisons and across all tested stimulus conditions, while the stricter `cross < both within` signature is fully met only in the supplementary cross-area pair and remains partial in the two primary cross-depth pairs because one within-plane estimate is unusually weak or unusually strong.

This pattern matters for interpretation. If one demanded the strict bilateral criterion in every pair, the `000336` result would be mixed rather than uniformly confirmatory. Our retained claim is therefore narrower: the full analyzed bundle supports reproducibly structured above-null inter-plane coupling, while the strongest clean access-constraint match appears in the supplementary cross-area pair and the primary cross-depth pairs remain partial because of within-plane asymmetry. Those cross-depth partials do not immediately falsify H3, because the observed pattern remains structured and above null rather than collapsing toward indiscriminate or null-like coupling; the more immediate alternative interpretation is asymmetric within-plane composition or sampling rather than absence of cross-plane structure.

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, left, center, center, center, center, center, left),
  table.header(
    [*Pair*], [*Geometry*], [*ROIs (A/B)*], [*Cross r*], [*z vs null*], [*Within A*], [*Within B*], [*H3 verdict*],
  ),
  [pair_a], [cross-depth], [6 / 62], [0.0197], [4.98], [0.0169], [0.0377], [partial],
  [pair_b], [cross-depth], [4 / 25], [0.0295], [4.04], [0.1160], [0.0261], [partial],
  [pair_c], [cross-area], [27 / 53], [0.0224], [5.16], [0.0275], [0.0434], [positive],
)

#par(first-line-indent: 0pt)[
  #emph[Table 2. Summary of the DANDI `000336` full-bundle spontaneous coupling result. All three within-session pairs show cross-plane coupling above the circular-shift null. Only the supplementary cross-area pair cleanly satisfies `cross < both within`, whereas the two cross-depth pairs remain partial because of within-plane asymmetry.]
]

Condition-level coupling across the full analyzed bundle is shown in @fig-open-data-000336-coupling-by-condition.

#figure(
  image("figures/figure_8_open_data_000336_coupling_by_condition.png", width: 98%),
  caption: [Open-data evaluation of full-bundle DANDI `000336` by condition. Across all three analyzed bundle pairs, cross-plane coupling remains above the circular-shift null in every plotted condition family. In the two primary cross-depth pairs, the stricter `cross < both within` criterion is only partially met because one within-plane population is unusually weak or unusually strong, whereas the supplementary cross-area pair provides the cleanest bilateral access-constraint match. The figure therefore supports a bounded access-constraint claim across the full available bundle rather than a hand-picked subject subset.]
) <fig-open-data-000336-coupling-by-condition>

To ask whether perturbing a candidate postsynaptic structural-write mechanism selectively weakens cross-day stabilization while sparing broader within-session structure, we turned to DANDI `001710`. The analysis was extended from the earlier four-subject bridge to a broader `23`-subject genotype bundle (`7` Cre, `9` Ctrl, `7` SparseKO; `139` NWB files total) @Plitt2026DANDI001710. The dataset targets hippocampal place-code plasticity under perturbation of a postsynaptic membrane-fusion component, making the SparseKO subjects relevant as a putative structural-write disruption group rather than as an unspecified control line. Subject-level cross-day summaries retained `4` to `6` usable days per subject and yielded group means of `0.3374` for Cre, `0.2926` for Ctrl, and `0.2623` for SparseKO. Under the implemented subject-level permutation null, the main result is that SparseKO lay below Cre (`obs_diff = -0.0751`, `z = -2.1495`, `p = 0.009`), whereas the comparison against Ctrl was directionally similar but weaker (`obs_diff = -0.0303`, `z = -1.2856`, `p = 0.099`). Lag profiles remained lower for SparseKO across lags `1` to `5`, and the long-lag gap widened at lag `5` (`0.1973` in SparseKO versus `0.3634` in Cre and `0.2785` in Ctrl). The canonical group comparison uses `ch0` as a predefined one-channel-per-animal bookkeeping rule rather than as a claim that `ch0` is uniquely privileged biologically.

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
  #emph[Table 3. Summary of the broadened DANDI `001710` genotype bundle. Group means are computed from subject-level off-diagonal similarity summaries, with individual subjects contributing `4` to `6` usable days. SparseKO is lowest on the canonical first-channel pass and falls below Cre under the implemented subject-level permutation null.]
]

A dedicated robustness package also shows that the `001710` result is not completely channel-invariant. Across SparseKO subjects, channel `1` was on average somewhat more cross-day stable than channel `0` (`0.3296` versus `0.2623`) and slightly more arm-separated (`0.3119` versus `0.2796`), while both channels retained high split-half reliability. This means the `001710` effect is sensitive to channel choice: using `ch1` alone would materially attenuate the main group contrast, whereas the predefined `ch0` pass preserves the clearest one-channel-per-animal comparison while avoiding duplicate subject counts. The retained claim should therefore remain bounded: `001710` now supports a broadened subject-level cross-day stabilization deficit in SparseKO, strongest relative to Cre, but channel sensitivity and partial-session heterogeneity argue against treating the dataset as a closed-form proof of a unique structural-write mechanism.

The open-data result set should therefore be read as uneven but useful. H1 is positive yet modest relative to the underlying burst baseline. H3 is cleaner and currently more article-stable, but it still speaks only to the plausibility of structured fast access rather than to the full slow-write mechanism. H2 now has a materially stronger bridge in `001710`: the remapping-scale signal extends across multiple subjects per genotype and is null-separated relative to Cre, even though the comparison against Ctrl is weaker and the more selective arm-conditioned analysis remains future work.

A compact full-bundle summary of the spontaneous `000336` result across all analyzed pairs is provided in @fig-open-data-000336-summary, making the pair-level consistency visible without implying a separate replication cohort.

#figure(
  image("figures/figure_9_open_data_000336_replication.png", width: 88%),
  caption: [Full-bundle summary for the spontaneous DANDI `000336` signature. All three analyzed bundle pairs show positive cross-plane coupling above the circular-shift null. Only the supplementary cross-area pair cleanly satisfies `cross < both within`, whereas the cross-depth pairs remain partial because of within-plane asymmetry. The figure therefore emphasizes that the above-null H3 signature survives across the full available bundle without erasing the difference between partial and full access-constraint evidence.]
) <fig-open-data-000336-summary>

= Discussion

Taken together, the unified paper now supports a three-level claim structure. At the biological level, the strongest current evidence supports active dendrites, spine-level compartmentalization, local tagging and translation, replay-linked consolidation, and dendritic memory allocation as a coherent background for a slow accessibility hypothesis @London2005DendriticComputation @Spruston2008PyramidalNeurons @Larkum2009Synaptic @Major2013ActiveDendrites @Tonnesen2014 @Rogerson2014SynapticTaggingAllocation @Hacisuleyman2024DendriticTranslation @Kastellakis2023DendriticEngram. At the executable level, the theory admits a disciplined branch-resolved implementation whose strongest signatures are not reproduced by simpler baselines. At the empirical level, open datasets do not reveal the slow field itself, but they do constrain which observable consequences survive reproducible measurement. `000718` addresses offline linking, full-bundle `000336` addresses structured access constraints, and `001710` now provides a broader subject-level perturbation bridge for the context-sensitive retrieval branch.

This integrated view also clarifies what has been narrowed. The paper should not imply that all context sensitivity is evidence for slow structural writing; fast contextual and circuit gating can explain part of that story already @WangYang2018Routing @Keller2020ContextualModulation @Bos2025GainModulation. Nor should the `000718` result be overstated: it is best read as a modest enrichment effect above a strong event-driven population baseline, not as direct replay-sequence proof. Likewise, `001710` should not be overstated even after the broader bundle pass: the SparseKO deficit is now anchored across `23` subjects and is null-separated relative to Cre, but the comparison against Ctrl is weaker, the arm-label audit remains indirect, and channel-sensitive differences within SparseKO caution against treating the effect as fully genotype-pure. A soma-resolved synaptic-weight account could in principle reproduce some of the same macroscopic observables. What keeps the present branch-resolved interpretation useful is not that the open data uniquely identify a latent $M_b$ field, but that the same slow-accessibility variable offers one mechanistic explanation for the joint profile of replay-linked linking, structured inter-plane coupling, and selective cross-day fragility under a write-related perturbation without fitting each signature separately. Finally, speculative or excessively microtubule-specific claims are not required for the main argument. The strongest version of the unified manuscript is the one that keeps the structural-accessibility layer biologically serious, executable, and empirically constrained without pretending that its molecular implementation is already settled.

== Alternative interpretations and discriminating future tests

Several simpler explanations remain viable for parts of the observed signature profile. Soma- or synaptic-weight-centered models could reproduce some macroscopic observables without positing a slow branch-specific write variable. Fast contextual gating can already explain part of the disambiguation story. Dendritic allocation without persistent structural accessibility could explain short-range overlap and some memory-linking effects. In the `001710` perturbation bridge, a more general plasticity or stability impairment could also reduce cross-day similarity without uniquely targeting a structural write process.

The discriminating experiments are therefore branch resolved rather than purely behavioral. Stronger support for the present framework would come from longitudinal tracking of identified dendritic segments across linked memories, perturbations that dissociate fast contextual gating from slower replay-dependent stabilization, and interventions showing that cross-session linking degrades more strongly than within-session recall under structural disruption. The central prediction is not merely that dendrites matter, but that a persistent branch-specific accessibility bias should leave a selective signature that simpler alternatives cannot match jointly.

== Pathology and State Implications

Anesthetics bind directly to tubulin and dampen cytoskeletal dynamics @Craddock2012AnestheticBinding @Zizzi2022AnestheticTubulin @Kelz2019AnesthesiaBiology @Eckenhoff2001AnesthesiaBinding. More broadly, anesthetic loss of consciousness has been linked to disrupted apical and dendritic function in pyramidal neurons @Phillips2018Apical. Within the present framework, the most conservative interpretation is that anesthesia can transiently impair branch-level accessibility: dendritic locks fail to open reliably, structural support for gain is reduced, and memory-relevant routing becomes less effective. The point is not that anesthesia has already been shown to switch off a specific microtubular memory mechanism, but that anesthetic perturbation plausibly alters dendritic integration states and may secondarily disrupt the slower structural conditions that support context-sensitive access.

Microtubule-modulating drugs have also been shown to alter sensitivity to volatile anesthetics, consistent with a functional link between cytoskeletal dynamics and state-dependent accessibility @Li2025IsofluraneMT @Khan2024Microtubule. Particularly relevant here, sevoflurane induces Tau-phosphorylation-dependent enlargement of dendritic spine heads in mouse hippocampal neurons — an effect absent in Tau-knockout neurons — directly implicating the same Tau-dependent cytoskeletal pathway proposed as a candidate write mechanism for the slow structural field @Yan2025SevofluraneTau. This raises the possibility that anesthetics may transiently disrupt not only fast access states but also the structural write process itself. The model then generates a specific falsifiable prediction: sub-threshold cytoskeletal perturbations should preferentially impair cross-session memory linking relative to within-session single-trace recall, because linking depends on replay-driven updating of $M_b$ while immediate recall relies more on fast dendritic access states. If confirmed, such a dissociation would also be consistent with consolidation-specific components of post-operative cognitive dysfunction.

In Alzheimer's disease and related tauopathies, Tau detachment and missorting weaken dendritic spines and destabilize microtubule organization @Zempel2010 @Zempel2013TauSpastin @Hoover2010TauSpines. In the present framework, this is best treated as an illustrative route by which structural accessibility may fail: branches could lose some of the slow support needed to preserve stable access to distributed traces, making retrieval less precise, less flexible, and more context fragile. The observation that microtubule stabilizers can rescue cognitive deficits and normalize microtubule dynamics in tauopathy models supports the broader idea that structural state contributes to memory accessibility, even if the exact branch-level mechanism remains to be resolved @Barten2012Hyperdynamic. These pathology examples should therefore be read as interpretive consequences of the framework rather than as decisive evidence that the full branch-level accessibility model has already been established in vivo.

Several limitations remain essential. None of the open datasets directly measure a slow cytoskeletal accessibility field. The executable model is phenomenological and compressed relative to biology. The H1 effect in `000718` is positive but modest relative to the underlying burst baseline. The H3 result in full-bundle `000336` now spans all analyzed bundle pairs but still belongs to a narrow dataset family. In `001710`, the broad bundle now supports subject-level genotype comparisons and completed null tests, but the arm-label audit remains indirect, some subjects contribute only partial day coverage, and SparseKO channel sensitivity complicates any single-number summary. These are not fatal weaknesses; they are precisely the boundary conditions that keep the theory program honest.

== Overall Interpretation

This paper has argued for a restricted but biologically motivated claim: slow branch-based accessibility is a coherent, executable, and empirically contactable hypothesis for associative memory, not a uniquely demonstrated biological mechanism. In this framing, dendritic branches and spines provide fast access states, whereas cytoskeletal, transport, metabolic, and related structural variables provide a slower background bias on which local pathways remain easiest to recruit over time.

The contribution is therefore not a claim that a microscopic cytoskeletal code has been demonstrated. It is a theory program organized across three levels. First, current biology already supports the ingredients needed for such a program: active dendrites, spine compartmentalization, local tagging and capture, branch-specific allocation, replay-linked consolidation, and structural remodeling. Second, an executable branch-resolved model shows that these ingredients can be combined into a coherent mechanism whose strongest signatures are not reproduced by simpler baselines. Third, open-data analyses show that some of those predicted signatures survive contact with real recordings: in `000718`, offline high-synchrony events carry a modest but consistent excess enrichment of NeutralExposure-defined core units; in full-bundle `000336`, cross-plane coupling is reproducibly above null across all analyzed pairs yet remains weaker than within-plane coupling; and in `001710`, the broader perturbation bundle shows the weakest cross-day stabilization in SparseKO, with the strongest separation appearing relative to Cre while within-session structure remains broadly preserved.

Those results support neither triumphalism nor dismissal. They narrow the viable claim space. The `000718` signal is positive but modest relative to the broader population-burst baseline. The full-bundle `000336` result is cleaner, but still belongs to a narrow dataset family and does not directly reveal the slow structural field itself. The `001710` bridge is now more informative because it extends across subjects and survives a targeted null relative to Cre, yet it remains bounded by channel sensitivity, indirect arm-label validation, and partial day coverage in some animals. The executable model remains phenomenological rather than molecularly complete. Taken together, however, the biology, simulations, and open-data signatures justify treating slow structural accessibility as a serious empirical hypothesis rather than as a purely metaphorical one. Final validation will require branch-resolved perturbation experiments, richer longitudinal measurements, stronger null-tested comparisons, and closer linkage between molecular manipulations and observable access dynamics. The present results are best understood as substantial first support for a coherent research program rather than as its completed confirmation.

The most important next step is therefore selective sharpening, not rhetorical expansion. The framework will stand or fall on whether branch-resolved perturbation, richer longitudinal dendritic measurements, and more selective observable bridges can distinguish a genuine slow accessibility field from explanations based on fast gating alone. That is the standard by which the present proposal should be judged, and it is also what makes it scientifically useful: it defines a falsifiable path forward for studying how dendritic structure and memory access may be linked.

= Materials and Methods

== Dendritic and Spine Accessibility as the Fast Access Layer

=== Active Dendrites and Branch-Specific Access
Associative memory is not expressed only at the soma or network level. In pyramidal neurons, dendritic branches operate as semi-autonomous computational subunits that can amplify or veto clustered synaptic input through local nonlinear events @London2005DendriticComputation @Spruston2008PyramidalNeurons @Sjostrom2008DendriticExcitability @Larkum2009Synaptic @Major2013ActiveDendrites. From the perspective developed here, these events are best understood as fast access states: they determine which local inputs are allowed to influence downstream spiking, plasticity, and recruitment into a distributed memory trace. Hippocampal and cortical work further indicates that such local access is context dependent and can participate in route selection, cue completion, and memory differentiation @Knierim2016Tracking @Miranda2024CA3Remapping.

=== Spine Neck State and Local Compartmentalization
Spines provide an additional level of control. Their neck geometry shapes both electrical isolation and biochemical trapping, thereby influencing whether synaptic input remains local, is amplified locally, or propagates to the parent dendrite @Tonnesen2014 @Araya2014 @Zecevic2023ElectricalPropertiesSpines. At the same time, structural plasticity at this level is strongly actin dependent, which is important for keeping the present model biologically balanced: the fast access layer is not microtubule-only, but is built from an actin-dominated spine apparatus that can still be biased by microtubule invasion and related transport processes @Hotulainen2010ActinSpines @Borovac2018ActinDynamicsSpines. In this paper we therefore treat dendritic or spine-level access states as metastable local conditions under which a branch or spine is more or less available to participate in memory encoding and recall. In mechanistic terms, this is better understood as a biophysical rheostat in local accessibility than as an all-or-nothing switch.

== Candidate Cytoskeletal Mechanisms of Structural Accessibility

Our proposal does not require a unique molecular trigger or a microtubule-only memory substrate. The slower accessibility layer can in principle be implemented by interacting cytoskeletal, transport, and metabolic variables at the level of dendritic branches and spines. The most conservative reading is therefore plural: actin-dominated spine remodeling, dynamic microtubules, MAP and PTM state, and branch-local organelle support together provide candidate mechanisms by which recent activity can leave a longer-lived accessibility bias.

=== Spine-Coupled Structural Remodeling
Spine neck geometry and actin-dependent structural plasticity strongly influence whether synaptic currents and biochemical signals remain isolated or become more effectively coupled to the parent dendrite @Tonnesen2014 @Araya2014 @Hotulainen2010ActinSpines @Borovac2018ActinDynamicsSpines. Dynamic microtubules can transiently invade active spines and support enlargement, trafficking, and local stabilization @Merriam2011Dynamic @Merriam2013.
- *Relevance:* The slow accessibility layer can be implemented partly through the persistence of local structural states that make some dendritic routes easier to amplify, stabilize, and revisit than others.

=== Transport and Metabolic Support
Plasticity depends not only on local geometry but also on whether cargo delivery, local translation support, and mitochondrial positioning can be sustained at a branch. Microtubule-based transport, dendritic mitochondrial parking, and postsynaptic compartment stabilization therefore provide a second mechanism family by which accessibility can remain uneven across branches @Faits2016 @ChangReynolds2006MitoTrafficking @Mironov2004MitoTransport @Misgeld2007DendriticMito @Rangaraju2019MitoCompartments @Thomas2023PostsynapticMito @Bapat2024VAP.
- *Relevance:* Structural accessibility can be expressed as differential readiness for energetically expensive or biosynthetically demanding plasticity, not only as a geometric gate.

=== Tau-, MAP-, and PTM-Dependent Stabilization
Tau and other MAP-linked processes regulate microtubule stability, exchange, and cross-talk with actin, while PTM state can influence whether local cytoskeletal organization remains permissive to transport and remodeling @Dent2017MicrotubulesMemory @Dent2020DynamicMicrotubulesSynapse @Elie2015TauCoOrganizes @Rosenberg2008TauBinding @Kadavath2015TauLattice @Biswas2024TauExchange.
- *Relevance:* In the present framework these factors are best treated as candidate mechanisms for maintaining or rewriting slow branch-level accessibility, not as direct carriers of memory content.

== Mechanistic Routes from Structural State to Fast Access

How can a slow structural state bias a fast memory-relevant dendritic event? We emphasize two conservative access-control routes, followed by a slower write mechanism in the next section.

=== Local Geometry and Impedance Control
The most direct interface is structural modulation of the dendritic spine and its coupling to the parent branch.
- *Mechanism:* NMDA receptor activation drives actin polymerization, while transient microtubule invasion can support spine enlargement and local cargo delivery @Merriam2013 @Tonnesen2014 @Araya2014.
- *Gating Logic:* Changes in spine head and neck geometry alter effective neck resistance ($R_("neck")$), thereby modulating how strongly synaptic input influences branch-level integration.
    - *Structurally supported state:* enlarged or stabilized spine geometry lowers effective impedance and facilitates propagation into the dendritic branch.
    - *Structurally deprived state:* thinner or destabilized spine geometry raises effective impedance and favors local isolation or failure of propagation.
In this sense, structural accessibility acts as a *biophysical rheostat* beneath faster dendritic events.
This mechanistic contrast is illustrated in @fig-structural-impedance-gating.

#figure(
  image("figures/figure_1_structural_impedance_gating_v3.png", width: 90%),
  caption: [Structural impedance gating at the spine level. (A) In a low-access structural state, reduced cytoskeletal support is associated with a thinner spine neck and higher effective impedance, reducing propagation from the synapse to the parent dendrite. (B) In a high-access structural state, activity-dependent structural support enlarges the spine compartment, lowers effective impedance, and facilitates current flow into the dendritic branch. The figure illustrates the paper's core mechanistic claim that slow structural state can bias fast synaptic-to-dendritic access without itself carrying mnemonic content.]
) <fig-structural-impedance-gating>

=== Transport and Metabolic Readiness
Sustaining high-frequency firing imposes a massive metabolic cost. Gamma oscillations (30-80 Hz) are uniquely vulnerable to metabolic impairment, disappearing before slower rhythms when mitochondria are inhibited @Kann2011 @Kann2012EnergyDemandReview @Whittaker2011MitoGamma.
- *Mechanism:* Dendritic mitochondria are mobile but "park" near active synapses, and this positioning depends on cytoskeletal transport plus activity-dependent stopping rules @Faits2016 @ChangReynolds2006MitoTrafficking @Mironov2004MitoTransport @Misgeld2007DendriticMito.
- *Gating Logic:* Cytoskeletal organization, together with scaffold proteins that stabilize postsynaptic mitochondrial compartments @Rangaraju2019MitoCompartments @Thomas2023PostsynapticMito @Bapat2024VAP, helps define where energetic and biosynthetic support can be maintained. If that support is weak, a branch may be less able to participate reliably in high-frequency synchrony or consolidation.

== The Cytoskeletal Learning Cycle

How does the system learn? We propose a three-loop model in which cytoskeletal remodeling contributes to the slow writing of branch-level accessibility structure.
The proposed three-loop write logic is summarized in @fig-cytoskeletal-learning-cycle.

#figure(
  image("figures/figure_2_Cytoskeletal learning cycle_v2.png", width: 85%),
  caption: [The cytoskeletal learning cycle. Fast dendritic and synaptic activity generates local tags and calcium-dependent eligibility signals at specific branches and spines. During later consolidation windows, including replay- and sleep-associated periods, these transient signals are converted into slower structural updates of the accessibility field. In this view, cytoskeletal remodeling does not replace synaptic plasticity, but contributes a slower write process through which branch-level accessibility becomes progressively stabilized.]
) <fig-cytoskeletal-learning-cycle>

=== Tau-Dependent Remodeling as a Candidate Write Mechanism
We treat *Tau-dependent remodeling* as one plausible candidate mechanism by which information could be written into the slow structural accessibility field.
- *Mechanism:* Tau stabilizes microtubules. When phosphorylated (for example by kinases such as CaMKII), Tau detaches, making the MT lattice less stable and more permissive to reorganization; when dephosphorylated, it binds and can help re-stabilize the lattice @Zempel2010 @WangMandelkow2013TauReview. Here, Tau-dependent remodeling is treated as one candidate implementation of slower structural rewriting, not as a uniquely established molecular write head.
- *Pathology as a clue:* In Alzheimer's disease and related conditions, hyperphosphorylation causes Tau to missort into dendrites and invade spines, weakening synapses and altering spine geometry @Mitchell2025 @Hoover2010TauSpines. Within the present framework, this is useful less as proof of a completed write mechanism than as one biologically relevant route by which structural accessibility could fail.

=== Sleep-Associated Consolidation as a Candidate Structural Window
We treat NREM sleep spindles (12-15 Hz) as one plausible class of consolidation windows in which transient dendritic tags may be converted into more durable branch-level accessibility biases @Ulrich2016SleepSpindles @PeyracheSeibt2020Spindles. This interpretation is strengthened in stages rather than by a single decisive finding. First, spindle-rich NREM activity has long been associated with memory consolidation and plasticity-related processing @Ulrich2016SleepSpindles @PeyracheSeibt2020Spindles. Second, dendrite-targeted recordings during natural sleep indicate that spindle-rich oscillatory periods are accompanied by compartment-specific dendritic $"Ca"^(2+)$ activity, while sleep-circuit work suggests that nested slow-oscillation and spindle states create conditions favorable for dendritic plasticity @Seibt2017DendriticSpindles @niethard2018 @Niethard2021SleepCalcium. Third, sleep after learning is associated with branch-specific spine formation, clustered structural remodeling, and biosynthetically active engram reactivation, indicating that nocturnal consolidation involves local molecular rewriting rather than replay alone @Yang2014SleepSpines @Adler2021SleepFilopodia @Sun2020SleepPlasticity @Wang2024SleepDependentEngramReactivation @Hacisuleyman2024DendriticTranslation. Taken together, these results make it biologically reasonable to treat spindle-rich sleep as a candidate window in which recent local tags could be converted into more durable structural accessibility states @ReyesResina2021SleepConsolidation.

Accordingly, the strongest current version of the argument is not that sleep spindles have already been shown to rewrite a single cytoskeletal subsystem directly, but that spindle-rich sleep provides a biologically plausible consolidation milieu linking dendritic calcium signaling, local translation, synaptic restructuring, and longer-timescale stabilization. The specific role of Tau-dependent or broader cytoskeletal remodeling within that milieu remains a target for empirical testing @ReyesResina2021SleepConsolidation @Uchida2014LearningMT.

=== Cytoskeletal Plasticity: The Credit Assignment Mechanism
A central challenge is explaining how branch-specific structural change could depend jointly on recent local activity and later behavioral relevance. To make that problem explicit, we use a phenomenological three-factor learning rule in which a local eligibility trace is combined with a delayed modulatory signal, analogous to Synaptic Tagging and Capture (STC) mechanisms and neoHebbian three-factor learning rules @fremaux2016neuromodulated @RedondoMorris2011STC @Gerstner2018EligibilityTraces.

The canonical coarse-grained update for the slow structural field is:

$ dot(M)_b(t) = eta E_b(t) sigma(delta(t) - theta_delta) (1 - M_b(t) / M_"max") - lambda_M M_b(t) + sqrt(2 T_("eff")) xi_b(t) $

Where:

1.  *$E_b(t)$ (The Eligibility Trace / "The Tag"):*
    A localized, decaying memory of recent high-frequency activity in dendritic branch $b$.
    - *Biophysics:* Driven by local Calcium influx ($"Ca"^(2+)$) and CaMKII activation.
    - *Function:* It marks specific branches or spines as eligible for later consolidation, but does not by itself trigger durable rewriting.

2.  *$delta(t)$ (The Instructional / Salience Signal):*
    A delayed neuromodulatory or systems-level factor indicating that recently tagged activity should be consolidated.
    - *Biophysics:* Candidate carriers include phasic Dopamine, Acetylcholine, or related salience-linked signals @Yagishita2014DopamineTiming.
    - *Function:* The thresholded term $sigma(delta - theta_delta)$ acts as a write-enable gate rather than as a full signed error code.

3.  *The Bounded Consolidation Term:*
    The factor $(1 - M_b / M_"max")$ expresses finite local consolidation capacity.
    - *Interpretation:* Recently tagged branches are preferentially stabilized when instructional signals arrive, but the update saturates as local accessibility approaches capacity.

4.  *$lambda_M$ and $T_"eff"$ (Turnover and Structural Noise):*
    The decay term represents ongoing turnover or destabilization, while the noise term captures stochastic remodeling in an active cytoplasmic environment @fodor2016 @needleman2017.

This rule is a phenomenological instantiation of neuromodulated STC-like learning @fremaux2016neuromodulated @RedondoMorris2011STC @Gerstner2018EligibilityTraces rather than a detailed biochemical rate model. Importantly, the minimal rule is intentionally asymmetric: it models selective stabilization of tagged branches, while weakening, differentiation, or unlinking arise through finite capacity, decay, competition for local capture resources, and failure to refresh the trace, not through an explicit signed anti-learning term. Richer signed update rules remain possible future extensions, but they are not required for the present manuscript's core claim.

== Minimal Biological Formalization

We formalize the interaction between fast dendritic integration and slow structural accessibility without assuming a full consciousness-level manifold model. The goal is to give a restrained language for the biological hypothesis.

=== Notation
- $x_b(t)$: fast integration state of dendritic branch $b$ (e.g., local voltage, plateau propensity, or branch-level activation).
- $s_i(t)$: local spine or synapse accessibility state, summarizing neck coupling, structural stability, and local efficacy.
- $M_b(t)$: slow structural state of branch $b$, including cytoskeletal organization, PTM state, and transport/metabolic support.
- $E_b(t)$: eligibility trace or local tag for branch $b$.
- $A_b(t)$: effective accessibility of branch $b$, jointly shaped by fast local state and slow structural bias.
- $a_(mu b)$: branch-allocation weight indicating how strongly trace $mu$ depends on branch $b$.
- $R_mu(t)$: recall-support for memory trace $mu$, representing how strongly the current accessibility pattern favors recruitment of the trace.
- $tau_"fast" << tau_"slow"$: timescale separation between fast dendritic/synaptic dynamics and slower structural remodeling.

=== Fast Access Dynamics

At the fast timescale, we treat dendritic access as depending jointly on input, local spine state, and slow structural bias. To make the paper's core claim explicit, we introduce an effective accessibility factor:

$ A_b(t) = A_b^f(x_b(t), s(t), C(t)) A_b^s(M_b(t)) $

where $C(t)$ denotes optional fast contextual or circuit-level gating. The factorization is schematic rather than literal, but it captures the intended interpretation: fast dendritic and circuit dynamics determine momentary opening, while the slow structural state determines which branches remain persistently easier to recruit. Multiplication is used here as a minimal gain-like approximation: the slow term rescales how effectively fast access can be expressed without separately introducing additional thresholds or interaction parameters. Alternative formulations, including additive threshold shifts or nested sigmoidal gates, remain plausible and may differ quantitatively, but they share the same qualitative prediction that slower structural state changes how readily a branch can enter a high-access regime.

We then write

$ dot(x)_b = A_b(t) F_b(x, I, s) $

where $I$ denotes synaptic drive. In this language, instantaneous accessibility determines whether a dendritic subunit can participate in amplification, propagation, and plasticity, whereas $M_b$ contributes a slower dynamic bias over how readily that access state is recruited.

We summarize spine-level accessibility by:

$ dot(s)_i = G_i(s, x, M) $

where structural dilation, local biochemical state, and branch-specific context can all modify whether a synapse remains electrically isolated, chemically privileged, or strongly coupled to the parent branch @Tonnesen2014 @Araya2014.

=== Slow Structural Update

The slower accessibility field evolves according to bounded, tag-dependent consolidation:

$ dot(M)_b(t) = eta E_b(t) sigma(delta(t) - theta_delta) (1 - M_b(t) / M_"max") - lambda_M M_b(t) + sqrt(2 T_("eff")) xi_b(t) $

where $E_b$ marks recently active branches, $delta(t)$ is a neuromodulatory or systems-level instructional signal, the bounded term captures finite local consolidation capacity, and the decay term captures ongoing turnover or destabilization. This equation should still be interpreted as a minimal phenomenological write rule, not a literal biochemical rate law. In this form it primarily models selective stabilization and turnover; weakening, separation, or unlinking arise through differential refresh, finite capacity, and decay rather than through an explicit signed anti-learning term. Richer capture variables for local translation or replay can be added in an appendix-level extension.
The formal relationships among fast access, slow structural bias, and retrieval support are summarized in @fig-minimal-biological-formalization.

#figure(
  image("figures/figure_3_minimal_biological_formalization_v3.png", width: 90%),
  caption: [Minimal biological formalization of fast access and slow structural bias. Fast branch integration states ($x_b$), local spine-access states ($s_i$), and effective branch accessibility ($A_b$) define the moment-to-moment opening of dendritic routes, whereas the slow structural state ($M_b$) and local eligibility traces ($E_b$) determine which branches remain persistently easier to recruit and consolidate. Retrieval support for a trace ($mu$) depends on whether the branches on which that trace depends are both active and accessible at recall. The figure summarizes the paper's central formal move: memory participation depends on a slow structural gate beneath faster dendritic and circuit dynamics.]
) <fig-minimal-biological-formalization>

=== Recall Support and Structured Accessibility

We model memory retrieval as depending on whether current branch states align with the accessibility requirements of a stored trace:

$ R_mu(t) = sum_b a_(mu b) A_b(t) x_b(t) $

where $a_(mu b)$ represents how strongly trace $mu$ depends on branch $b$. In this view, associative memory is not determined by synaptic weights alone; it is modulated by a slowly learned accessibility landscape distributed across dendritic branches. A trace can fail to express not because its synapses are absent, but because the relevant branch subset is insufficiently accessible at the time of retrieval.

This quantity should be read as a branch-level recall support, not yet as a full attractor readout. A downstream nonlinearity or recurrent completion step can transform this support into expressed retrieval, for example through $hat(y)_mu(t) = Phi(R_mu(t) - theta_mu)$, where $Phi$ is a thresholding or sigmoidal readout.
The branch-allocation intuition behind linking and contextual retrieval is illustrated in @fig-memory-linking-contextual-retrieval.

#figure(
  image("figures/figure_4_memory linking and contextual retrieval_v2.png", width: 90%),
  caption: [Branch allocation as a mechanism for memory linking and contextual retrieval. Memories formed close in time can preferentially recruit overlapping dendritic segments that remain in relatively high structural accessibility states. Under this view, later recall of one trace can partially facilitate recruitment of another, not only through cellular ensemble overlap but also through shared branch-level allocation. More independent memories rely on less-overlapping branch subsets and therefore remain more separable at retrieval.]
) <fig-memory-linking-contextual-retrieval>

=== Caveats and Modeling Choices

Four modeling choices and their limitations are worth keeping in view:

1.  #strong[Accessibility, not explicit content coding:]
    The model concerns which branches and spines are available to participate in encoding and recall. It does not claim that cytoskeletal variables alone represent mnemonic content.
2.  #strong[Phenomenological branch variables:]
    The variables $x$, $s$, $M$, and $R$ compress many biophysical processes into coarse-grained states. They are biologically informed summary variables rather than directly measurable single-molecule quantities.
3.  #strong[Factorized access is schematic:]
    The separation of $A_b$ into fast and slow terms is a compact way of stating the paper's central hypothesis. It does not imply that all relevant mechanisms are cleanly separable in vivo.
4.  #strong[Future geometric interpretation is optional:]
    The present formulation is compatible with later geometric abstraction, but such an abstraction is not required for the biological memory hypothesis to be meaningful or testable.

== Executable Model and Open-Data Pipelines

The executable results reported in Table 1 were produced by a branch-resolved Python simulator (`cytodend_keylock`) whose source code and experiment scripts are maintained in the public `cytodendaccessmodel` repository and archived in the manuscript-matched Zenodo snapshot. The canonical simulator uses a four-branch architecture with one overlap branch shared between the two traces, replay-dependent consolidation, and comparator baselines that remove or replace slow structural writing. Full parameter tables, baseline definitions, and reproducibility details are provided in `S2 Appendix`.

The open-data analyses were designed as fixed downstream signature tests of the structural-accessibility framework rather than as direct microscopic assays of a slow write variable. DANDI `000718` was analyzed with cross-session ROI registration, ensemble extraction, and event-versus-inter-event enrichment scoring. DANDI `000336` was analyzed with paired-plane coupling estimates against within-window circular-shift nulls. DANDI `001710` was analyzed with subject-level place-code observables, genotype-level permutation nulls, and channel-sensitive robustness checks. Full pipeline details and extended QC summaries are provided in `S3 Appendix`.

Because the open-data program spans multiple datasets and related sub-analyses, the reported p-values should be interpreted as analysis-specific anchors rather than as a single familywise-confirmatory survey across the entire paper. We use them to bound claims within each dataset-specific pipeline, not to assert one globally corrected significance statement across all open-data results.

== Statistical analysis and reporting conventions

Analyses were run in Python `>=3.10` using the archived `cytodendaccessmodel` codebase (`0.1.0`) and its declared scientific dependencies, including `numpy`, `pyyaml`, `torch`, `dandi`, `pynwb`, and `matplotlib` for visualization. The manuscript-matched software record is defined by `pyproject.toml` in the Zenodo snapshot. The executable simulator is deterministic in its canonical configuration unless a robustness run explicitly enables noise or random drift; where a seed is required, it is stated in `S2 Appendix`.

For empirical null-separation summaries, the nominal alpha level was `0.05`. The reported tests are analysis specific rather than familywise across the entire paper. In `000718`, the main quantities are event-versus-inter-event enrichment deltas and registration-shuffle contrasts across `3` NeutralExposure-to-offline session pairs; threshold sweeps are sensitivity analyses rather than separate confirmatory endpoints. In `000336`, the main quantities are cross-plane `r`, within-plane `r`, and `z` versus within-window circular-shift null across `3` within-session pairs (`6` NWB files total). In `001710`, the main inferential step is a subject-level permutation null on genotype contrasts using `23` subjects (`7` Cre, `9` Ctrl, `7` SparseKO) drawn from `139` NWB files.

The `001710` genotype contrasts are evaluated directionally for the planned ordering `SparseKO < comparator`; the `000718` and `000336` z-versus-null summaries are likewise directional tests of positive excess enrichment or positive above-null coupling rather than symmetric two-sided screens for any difference. No global multiple-testing correction was applied across all datasets because the three dataset families address different hypotheses and use different null constructions; instead, claims are bounded within each pipeline and robustness analyses are reported separately in `S3 Appendix`. Exclusions followed pipeline-specific QC and data-availability rules rather than outcome-driven filtering: short `000336` stimulus fragments were merged or omitted if they failed the minimum block-duration rule, and `001710` subjects contributed `4` to `6` usable days depending on recording coverage. Missing data were handled conservatively by dropping unavailable sessions, windows, or days from the relevant summaries rather than imputing them.

Effect sizes are reported throughout as enrichment deltas, coupling coefficients, null-separated `z` scores, and subject-level mean differences. Because several analyses rely on empirical nulls, small bundles, or both, the manuscript emphasizes these effect sizes together with pipeline-specific null-separation summaries rather than forcing a single asymptotic confidence-interval style across all result families.

= Data and Code Availability

All author-generated code, analysis scripts, configuration files, figures, and manuscript-source material necessary to inspect and reproduce the analyses reported in this study are publicly available in the `cytodendaccessmodel` GitHub repository (`https://github.com/NoeticDiffusion/cytodendaccessmodel`) and in the manuscript-matched archived snapshot at Zenodo (DOI `10.5281/zenodo.19498499`; `https://doi.org/10.5281/zenodo.19498499`). The code is released under the GNU General Public License v3.0 (`GPL-3.0`). The empirical analyses use openly available third-party datasets from the DANDI Archive (`000718`: `https://dandiarchive.org/dandiset/000718`, `000336`: `https://dandiarchive.org/dandiset/000336`, `001710`: `https://dandiarchive.org/dandiset/001710`), which are not redistributed by the author and should be accessed through their original archive records. Derived manuscript-facing outputs are organized under `data/dandi/triage/000718`, `data/dandi/triage/000336`, and `data/dandi/triage/001710`. Primary repository entry points for reproduction are organized under `experiments/`, `configs/`, `src/dandi_io/`, and `src/dandi_analysis/`, with dataset-specific QC and robustness artifacts summarized in `S3 Appendix`.

= Acknowledgments

The author gratefully acknowledges editorial and analytical assistance from large language models during literature synthesis, structured critique, drafting, and revision. These tools are not authors and did not provide peer review.

#bibliography("references_cytoskeletal_dendritic_accesibility_model.bib")

#pagebreak()

= Supporting Information Captions

S1 Appendix. Mathematical framework and executable bridge. Extended formalization of the structural-accessibility model, including compact state-space equations, factorized accessibility, optional resource-capture extensions, and the theory-to-executable mapping used to relate the biological formalism to the simulator.

S2 Appendix. Executable simulator architecture, parameters, and comparator baselines. Detailed simulator architecture, canonical parameter set, encoding and consolidation protocol, baseline definitions, and stochastic reproducibility notes for the branch-resolved model.

S3 Appendix. Open-data pipelines and reproducibility details. Dataset-specific pipeline descriptions, robustness summaries, subject-level QC notes, null tests, and repository entry points for DANDI `000718`, `000336`, and `001710`.

Associated supporting-information files: `S1_Appendix.pdf`, `S2_Appendix.pdf`, and `S3_Appendix.pdf`.
