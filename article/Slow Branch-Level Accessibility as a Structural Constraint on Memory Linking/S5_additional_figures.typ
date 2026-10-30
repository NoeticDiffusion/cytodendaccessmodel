#import "template.typ": essay-template

#show: doc => essay-template(
  short_title: [S5 Appendix],
  doc,
)

// Figures in this appendix are numbered S5.1, S5.2, … rather than 1, 2, …
#set figure(numbering: (..num) => "S5." + str(num.at(-1)))

= S5 Appendix. Additional conceptual figures

== S5.1. Biological motivation for a slow accessibility variable

The figure below situates $M_b$ as a phenomenological compression of slow branch-local
processes: spine-neck geometry, actin remodeling, dynamic microtubule entry, local
translation, transport readiness, and mitochondrial support. None of these processes
is required to map one-to-one onto $M_b$. Rather, they collectively motivate the class
of slow branch-local access constraints that $M_b$ represents.

The figure is provided for conceptual orientation. It does not represent a direct
biological measurement, a validated model of any specific molecule, or a claim that
cytoskeletal structures directly store memory content. The main article's claim rests on
simulator discrimination, not on the biological sketch.

#figure(
  image("figures3_concepts/biological_motivation.png", width: 95%),
  caption: [
    *Biological motivation for the slow accessibility variable $M_b$.*
    Fast branch opening is driven by cue input, contextual gating, and local
    dendritic/spine state (left side). Slow branch-local processes --- spine geometry,
    actin dynamics, dynamic microtubule entry, local translation, transport readiness,
    and metabolic support --- evolve on longer timescales and are compressed into the
    latent variable $M_b$ (right side).     $M_b$ is a phenomenological variable: it is
    biologically motivated and operationally defined, but it should not be read as a
    direct molecule count or as proof of a cytoskeletal memory code. Candidate processes
    include spine-neck compartmentalization @Tonnesen2014 @Araya2014 @Popovic2015,
    actin-dependent spine remodeling @Hotulainen2010ActinSpines @Borovac2018ActinDynamicsSpines,
    dynamic microtubule entry into spines @Merriam2011Dynamic @Merriam2013 @Kapitein2011_MAP2
    @Dent2017MicrotubulesMemory @Dent2020DynamicMicrotubulesSynapse, branch-local
    translation @Govindarajan2011DendriticBranch @Hacisuleyman2024DendriticTranslation
    @Das2023LocalTranslationMemory, and mitochondrial/metabolic support
    @Rangaraju2019MitoCompartments @Thomas2023PostsynapticMito @Bapat2024VAP.
  ],
) <fig-biological-motivation>

// ─────────────────────────────────────────────────────────────────────────────
== S5.2. Structural impedance gating at the spine neck

Microtubule-associated remodeling is one illustrative candidate contributor to
structural accessibility @Merriam2011Dynamic @Merriam2013 @hu2008activity
@Kapitein2011_MAP2 @Dent2020DynamicMicrotubulesSynapse. In the schematic below, a
low-support structural state is depicted as higher effective impedance at the spine
neck, limiting voltage spread from dendrite into spine @Tonnesen2014 @Araya2014
@Popovic2015 @Zecevic2023ElectricalPropertiesSpines. A high-support structural state is
depicted as lower effective impedance, permitting signal propagation. This contrast
illustrates how slow branch-local structural changes could gate signal flow
independently of fast synaptic input strength. The fast/slow access factorization in
the model is motivated by this class of slow-timescale process; $M_b$ compresses any
combination of them.

This figure is illustrative. The model does not require MT dynamics specifically;
$M_b$ may represent any combination of slow branch-local processes.

#figure(
  image("figures/figure_1_structural_impedance_gating_v3.png", width: 90%),
  caption: [
    *Structural impedance gating at the spine neck (schematic).*
    *(A)* Low-support state: the spine-neck structural configuration is schematized as
    higher effective impedance, limiting voltage spread from dendrite into spine.
    *(B)* High-support state: a structurally permissive configuration is schematized as
    lower effective impedance, permitting signal propagation. Microtubule-associated
    remodeling is one illustrative candidate contributor; the schematic does not assert
    that MT dynamics are the unique or dominant mechanism. $M_b$ compresses this and
    related slow-timescale branch-local processes into a single latent variable.
  ],
) <fig-impedance-gating>

// ─────────────────────────────────────────────────────────────────────────────
== S5.3. Minimal biological formalization: variable structure and equations

This historical schematic illustrates a broader theoretical formulation, not the discrete
updates executed in E017--E025. In particular, its $x_b$ is an ungated integration state
and its readout includes $A_b$ explicitly. In the implemented simulator, $x_b$ already
includes access gating, so recall support does not multiply by $A_b$ again. See S1,
Mapping from the continuous family to the executable simulator, for the limitations
of the correspondence and the exact discrete update equations.

#figure(
  image("figures/figure_3_minimal_biological_formalization_v3.png", width: 95%),
  caption: [
    *Historical theoretical formalization: variable structure and update equations.*
    This diagram is not the executed E017--E025 readout; see the explanatory text above.
    Left: fast states ($x_b$ branch integration, $s_t$ spine access, $A_b$ effective
    access). Centre: slow states ($M_b$ structural bias, $overline(E)_b$ eligibility tag,
    $P_b$ capture resource). Right: memory readout $R_mu = sum_b a_(mu b) A_b x_b$.
    Bottom left: fast factorization $A_b = A_f(x_b, s_t, C) dot.c A_s(M_b)$. Bottom
    right: slow update $dot(M)_b = eta overline(E)_b sigma(delta - theta)(1 - M_b /
    M_"max") - lambda_M M_b + xi(t)$. Interpretation note: synapses do not disappear;
    rather, the relevant branch subset can become less accessible.
  ],
) <fig-minimal-formalization>

// ─────────────────────────────────────────────────────────────────────────────
== S5.4. Memory linking and contextual retrieval via shared branches

This figure illustrates the key operative prediction: two episodes encoded close in
time that share a dendritic branch segment can become linked, so that retrieving one
helps recruit the other. An unrelated episode (Episode C) that does not share a branch
remains independent. This is the core testbed for the slow-accessibility hypothesis
and provides intuition for the simulator's two-trace overlap motif.

#figure(
  image("figures/figure_4_memory linking and contextual retrieval_v2.png", width: 90%),
  caption: [
    *Memory linking and contextual retrieval via shared dendritic branches.*
    Episodes A and B (encoded close in time, $t_1$ and $t_1 + Delta t$) share a branch
    segment (dashed box). Replay-dependent slow structural writing raises $M_b$ on the
    shared branch, increasing the linking score $L_(mu nu)$; retrieval of A can recruit
    B (linked recall, upper right). Episode C (independent, no shared branch) remains
    unlinked (lower right). This schematic motivates the two-trace overlap motif used
    throughout the simulator experiments and the generalized motif tests in E021--E021R.
    Memory linking via shared dendritic structure is supported by work on engram neurons,
    contextual linking, and hippocampal memory allocation
    @Choucry2024MemoryLinkingIdentity @Kastellakis2023DendriticEngram
    @Sehgal2025ContextLinking @Zaki2025Engram @Zaki2024OfflineLinking
    @Guskjolen2023EngramNeurons @Frankland2005RecentRemoteMemories.
  ],
) <fig-memory-linking>

// ─────────────────────────────────────────────────────────────────────────────
== S5.5. Evidence ladder and claim boundary

The figure separates historical internal mechanism diagnostics and boundary analyses from
the primary E025 common-probe comparison. The equal-budget, locked-holdout comparison
does not demonstrate a unique slow-writing advantage on its defined task. The evidence
is limited to the tested implementations, adapters, parameter ranges, and generated
topologies; S3 open-data bridges do not validate the latent state.

#figure(
  image("figures3_concepts/Fig_e025_s5_evidence_ladder.svg", width: 95%),
  caption: [
    *Evidence ladder and claim boundary.* E017--E022R characterize simulator dynamics,
    internal mechanism signatures, and parameter and motif boundaries. E024 develops a
    common probe; E025 compares tuned model families on a locked holdout. Hebbian and
    resource-only variants exceeded the full model on the primary readout, while shuffled
    replay was practically equivalent under the registered decision rule. No biological
    measurement of $M_b$ or unique molecular memory code is established.
  ],
) <fig-evidence-ladder>

// ─────────────────────────────────────────────────────────────────────────────
== S5.6. Overlap damage and targeted rescue logic

This figure summarizes the mechanistic logic behind SIG-D and SIG-E. Two traces share
an overlap branch that is selectively strengthened by replay-linked slow writing.
Focal damage to the shared branch disproportionately reduces linking relative to
single-trace recall (SIG-D). Overlap-targeted rescue restores linking more effectively
than generic rescue applied to non-overlap branches (SIG-E).

#figure(
  image("figures3_concepts/overlap_damage_and_targeted_rescue_logic.png", width: 90%),
  caption: [
    *Overlap damage and targeted rescue logic.*
    Two traces (A, B) share an overlap branch whose slow structural accessibility $M_b$
    has been raised by replay-dependent consolidation. Focal damage to the overlap branch
    selectively reduces the linking score $L_(A B)$ while largely preserving single-trace
    recall support $R_A$ and $R_B$ (SIG-D). Targeted rescue --- which restores eligibility
    on the overlap branch --- recovers linking more than non-targeted generic rescue
    (SIG-E). This is a historical mechanism-specific diagnostic; E025 uses a separate
    common-probe comparison and does not test this rescue protocol across all families.
  ],
) <fig-overlap-damage-rescue>

// ─────────────────────────────────────────────────────────────────────────────
== S5.7. Comparator evaluation and model-discrimination logic

E025 compares specified model families using a shared probe and readout after equal-budget
tuning. Historical E018/E022 mechanism signatures and proxy outcomes describe internal
behavior under their original protocols; they are not the E025 common-readout test and
cannot by themselves establish an output-level advantage for slow writing.

#figure(
  image("figures3_concepts/Fig_e025_s5_comparator_comparison.svg", width: 95%),
  caption: [
    *Comparator evaluation and model-discrimination logic.* E025 used a shared probe,
    model-defined adapters, 32 candidate configurations per family, and independent
    development, validation, and holdout topology draws. On the immediate readout the
    full model exceeded five testable alternatives, was practically equivalent to
    shuffled replay, and was exceeded by Hebbian and resource-only variants. Eligibility-only
    had no qualifying candidate. These are outcomes for the specified implementations and
    the registered decision rule, conditional on the sign-flip assumption; S2 reports the
    post-hoc bootstrap sensitivity and further limits.
  ],
) <fig-comparator-fairness>

#bibliography("references_slow_branch_level_accessibility.bib")
