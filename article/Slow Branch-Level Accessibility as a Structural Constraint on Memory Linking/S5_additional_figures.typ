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

This figure provides a visual companion to the S1 mathematical appendix. It shows the
fast-state variables ($x_b$, $s_t$, $A_b$), the slow-state variables ($M_b$, $E_b$,
$P_b$), the memory readout ($R_mu$), and the two key equations: the fast factorization
$A_b = A_f dot.c A_s$ and the slow update $dot(M)_b$. Readers who find the S1
state-space notation dense may find this visual layout a more accessible entry point.

The figure uses the general biological notation from S1; the main text uses the
executable simulator form where $P_b(t) W(t)$ makes the consolidation window explicit.
See S1 §Correspondence between general and executable forms for the alignment.

#figure(
  image("figures/figure_3_minimal_biological_formalization_v3.png", width: 95%),
  caption: [
    *Minimal biological formalization: variable structure and update equations.*
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

The staircase below summarizes the primary simulator-first evidence chain from traceable
dynamics to motif boundaries. It also makes the retained claim boundary explicit:
supplementary open-data analyses are exploratory bridges, whereas direct biological
validation and any unique molecular code claim remain outside the support provided by
this paper. Readers unfamiliar with the paper's overall architecture may find this figure
useful as an orientation before reading the Results.

#figure(
  image("figures3_concepts/evidence_ladder.png", width: 95%),
  caption: [
    *Evidence ladder and claim boundary.*
    The staircase summarizes the simulator-first evidence chain from traceable branch
    dynamics (E017) through canonical signatures (E018), robustness (E019--E020), motif
    generalization (E021--E021R), hard comparators (E022), and replay audit (E022R). The
    right side marks what the paper does _not_ support: direct biological validation of
    $M_b$, a unique molecular memory code, or a validated DANDI/open-data claim. The
    supplementary open-data analyses (S3 Appendix) are exploratory bridges, not primary
    evidence.
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
    (SIG-E). This logic motivates the two-tier rescue protocol used in all comparator
    evaluations.
  ],
) <fig-overlap-damage-rescue>

// ─────────────────────────────────────────────────────────────────────────────
== S5.7. Comparator fairness and model-discrimination logic

This figure makes explicit the two-level evaluation design used for all comparators. An
alternative model is not rejected merely because it lacks $M_b$. It is first evaluated
on output-level metrics (linking gain, damage sensitivity, recovery index) that are
computable regardless of mechanism. It is then evaluated on structural mechanistic
signatures that require $M_b$-like branch-specific writing. Only when both levels fail
does the alternative fall outside the joint profile.

#figure(
  image("figures3_concepts/comparator_fairness.png", width: 90%),
  caption: [
    *Comparator fairness and model-discrimination logic.*
    Alternatives are evaluated first on output-level linking and damage patterns (left
    column: B1, B3, B5 metrics) and then on whether they reproduce overlap-branch
    structural writing, rescue selectivity, and cross-scale specificity (right column:
    SIG-A, SIG-B, SIG-E, gSIG-A). Comparators are not dismissed merely for lacking
    $M_b$; they are rejected when they fail the full mechanistic and output-level joint
    profile. This design prevents circular evaluation and directly addresses the
    comparator-fairness objection.
  ],
) <fig-comparator-fairness>

#bibliography("references_slow_branch_level_accessibility.bib")
