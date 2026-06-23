// Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking
// Typst source — converted from draft.md (E023)
// Uses template_eLife.typ for layout conventions.

#import "template_eLife.typ": elife-template, abstract-block, supplement-box, note-line, clean-table

#show: elife-template.with(
  title: [Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking],
  authors: [Robin Langell],
  affiliations: [Independent Researcher],
  correspondence: [hello\@noeticdiffusion.com],
  contributor_notes: [
    #align(center)[
      #text(size: 10pt, style: "italic")[
        A simulator-first test of replay-dependent structural writing, specificity, and rescue
      ]
    ]
  ],
  abstract: [
    Dendritic branches are active computational compartments whose local state can shape whether synaptic input is amplified, stabilized, or later recruited. However, it remains unclear whether memory linking requires only fast contextual gating and synaptic strengthening, or whether recent activity can also leave a slower branch-level accessibility bias that constrains future reuse. Here we introduce a
    simulator-first framework for testing slow branch-level accessibility as a structural constraint on memory linking. The model separates fast branch access from a slower structural variable, $M_b$, updated by eligibility, replay, and consolidation support. In a canonical two-trace motif, replay-dependent slow writing produces overlap-branch strengthening, linking gain, perturbation-sensitive loss, and targeted
    rescue. These signatures are traceable in exported simulator dynamics and are not reproduced by nine tested simpler or alternative comparator classes. Robustness analyses show that the joint profile occupies a bounded write--replay regime rather than a single tuned point. Scaling and motif tests show that the mechanism generalizes across tested branch counts and motif classes, while weak overlap and hub-like universal overlap define expected boundary conditions. These results support slow branch-level accessibility as an executable, falsifiable model-discrimination hypothesis. They do not establish a unique molecular or cytoskeletal memory code.
  ],
)

// ─── Author Summary ──────────────────────────────────────────────────────────
#supplement-box([Author Summary])[
  Memories are often modeled as changes in synaptic weights or recurrent network states.
  This paper explores a complementary possibility: that dendritic branches may also carry
  slow access biases that affect which local routes remain available for later
  stabilization, linking, and recall. We do not claim that dendrites or cytoskeletal
  structures directly store memory content. Instead, we ask whether a slow branch-level
  accessibility variable improves a minimal simulator of memory linking.

  This article takes a simulator-first approach: beginning with a canonical
  branch-resolved model, exporting its internal traces, and testing whether
  replay-dependent slow structural writing is required to reproduce a joint profile of
  overlap-branch strengthening, linking gain, selective perturbation vulnerability, and
  targeted rescue. We then compare the
  full model against several simpler alternatives, stress the parameter regime, scale the
  model beyond four branches, and test motifs with weak, strong, chain, hub, and sparse
  overlap. The result is bounded: the model works in regimes with sufficient but not
  universal overlap, fails when overlap is too weak, and loses specificity in hub-like
  motifs. The value of the framework is therefore not proof of a cytoskeletal memory
  code, but a clearer and more falsifiable simulator for testing slow dendritic-access
  constraints.
]

#v(8pt)

// ─── Introduction ─────────────────────────────────────────────────────────────
= Introduction

Memory models often emphasize synaptic weights, recurrent dynamics, pattern completion,
and contextual gating as the main substrates for association and retrieval. Those
mechanisms remain central to hippocampal and cortical computation @Koch1999Biophysics
@Knierim2016Tracking. The question addressed here is narrower: whether recent activity
can also leave a slower branch-level accessibility bias that changes which dendritic
routes remain easier to stabilize, reuse, or relink later.

Dendrites and spines are not passive conduits. Active dendritic conductances,
branch-local nonlinearities, and spine geometry can strongly shape whether synaptic input
is amplified, compartmentalized, or locally integrated @Larkum2009Synaptic
@Major2013ActiveDendrites @London2005DendriticComputation @Tonnesen2014 @Popovic2015.
These observations motivate treating branch access as a control problem in its own
right, rather than only a problem of synaptic weight strength.

Slow stabilization mechanisms are also biologically plausible. Synaptic tagging and
capture, eligibility traces, replay-linked consolidation, and local translation all
support the idea that transient activation can be converted into delayed and spatially
structured persistence @RedondoMorris2011STC @Gerstner2018EligibilityTraces
@Das2023LocalTranslationMemory @Wang2024SleepDependentEngramReactivation. Branch-local
energetic and transport constraints further suggest that recent use could bias later
availability without requiring a single dedicated storage molecule
@Rangaraju2019MitoCompartments @Thomas2023PostsynapticMito @Bapat2024VAP.

Memory linking provides a particularly clear testbed because temporally nearby traces can
share compartments, replay can revisit overlap, focal perturbation can target shared
routes, and rescue can ask whether restoring overlap-branch access selectively restores
linking @Choucry2024MemoryLinkingIdentity @Kastellakis2023DendriticEngram
@Sehgal2025ContextLinking @Zaki2025Engram. These studies do not directly measure $M_b$.
They motivate the hypothesis class tested here: a slow branch-local accessibility
variable that complements fast gating and synaptic strengthening.

The gap is therefore specific. Existing explanations can often produce context
separation or partial linking through contextual routing and gain modulation
@WangYang2018Routing @Keller2020ContextualModulation @Bos2025GainModulation, fixed
overlap geometry, weight-based associative strengthening @Podlaski2025HighCapacity
@Koch1999Biophysics @Knierim2016Tracking, replay without persistent structural change,
random slow drift, or global gain. What has been missing is a minimal executable test
that compares these alternatives within one branch-resolved simulator and asks which
mechanisms reproduce the joint profile of overlap writing, linking gain, perturbation
sensitivity, rescue selectivity, and topology-dependent specificity.

In this study, we introduce such a computational model. The simulator separates fast
branch access from a slower phenomenological structural variable, $M_b$, updated by
eligibility, replay, and consolidation support. The main claim is limited:
replay-dependent slow branch-level writing reproduces the joint simulator profile more
specifically than the tested alternatives. The model does not establish a unique molecular or
cytoskeletal memory code.

// ─── Operational Definitions ─────────────────────────────────────────────────
= Operational Definitions

#note-line[*Branch accessibility*][
  How readily a dendritic branch can participate in encoding, consolidation, linking, or recall.
]
#note-line[*Fast access*][
  Momentary opening driven by cue input, local dendritic/spine state, and context.
]
#note-line[*Slow structural accessibility*][
  A persistent branch-level bias variable, $M_b$, that changes how easily a branch
  can be recruited later.
]
#note-line[*Slow structural writing*][
  Replay-, eligibility-, and consolidation-dependent updating of $M_b$. We use
  "writing" in a computational sense: persistent simulator-state updating that changes
  later accessibility, not direct writing of memory content into a molecule.
]
#note-line[*Memory linking*][
  Increased cross-trace facilitation between memory traces. In the simulator, linking
  is operationalized as shared structural accessibility between trace allocations.
]
#note-line[*Single-trace recall*][
  Support for retrieving one trace without requiring cross-trace facilitation.
]
#note-line[*Comparator baseline*][
  A model variant that removes, replaces, or scrambles one mechanism to test whether
  the full model's joint profile is specific.
]
#note-line[*Targeted rescue*][
  Post-perturbation restoration of overlap-branch eligibility or structural access,
  compared against generic or non-targeted rescue.
]
#note-line[*Mechanistic signature*][
  A simulator outcome that directly tests a proposed internal mechanism, such as
  overlap-branch $M_b$ writing (SIG-A, SIG-B, SIG-E).
]
#note-line[*Output-level metric*][
  A comparator-agnostic readout --- such as linking gain, recall support, damage
  sensitivity, or recovery index --- that can be computed even for models without $M_b$.
]
// ─── Biological Motivation ───────────────────────────────────────────────────
= Biological Motivation for a Slow Accessibility Variable

The slow variable $M_b$ is not introduced as an arbitrary extra state. It compresses a
family of plausible branch-local processes that evolve more slowly than fast cue gating:
spine-neck geometry, actin remodeling, dynamic microtubule entry, local translation,
transport readiness, and mitochondrial support @Araya2014 @Hotulainen2010ActinSpines
@Borovac2018ActinDynamicsSpines @Merriam2011Dynamic @Merriam2013
@Kapitein2011_MAP2 @Rangaraju2019MitoCompartments.

The simulator does not require any of these processes to map one-to-one onto $M_b$.
Instead, they motivate the class of slow branch-local access constraints that $M_b$
represents. In that sense, $M_b$ is a phenomenological latent variable: biologically
motivated, operationally defined, and deliberately narrower than a claim about direct
molecular memory storage.

This biological motivation is summarized in Supplementary Figure S5.1 (S5 Appendix).

// ─── Conceptual Framework ────────────────────────────────────────────────────
= Conceptual Framework

== Fast access and slow structural bias

The model separates fast and slow access. At the fast timescale, dendritic branches
respond to cue input, contextual state, and local branch or spine conditions. At the
slower timescale, branch-specific structural accessibility biases whether a branch
remains easier to stabilize and reuse after replay or consolidation.

This distinction matters because fast contextual gating can explain some
disambiguation effects without requiring slow structural writing. The claim tested here
is narrower: replay-dependent slow writing is needed for the full joint profile of
overlap-branch structural gain, durable linking, targeted rescue, and topology-dependent
specificity.

== Minimal formalization

The effective accessibility of branch $b$ is:

$ A_b(t) = A_b^f(t) A_b^s(t) $

where $A_b^f(t)$ is fast access and $A_b^s(t)$ is slow access derived from $M_b(t)$.

Fast branch activity follows:

$ tau_x dot(x)_b = -x_b + A_b(t)[I_b(t) + R_b(t)] $

where $I_b(t)$ is cue input and $R_b(t)$ is replay or recurrent recruitment.

Eligibility evolves as:

$ tau_E dot(E)_b = -E_b + phi(x_b, I_b) $

Branch-local consolidation support evolves as:

$ tau_P dot(P)_b = -P_b + rho_"replay" r_b(t) + rho_nu nu(t) $

Slow structural accessibility is updated by:

$ dot(M)_b = eta E_b(t) P_b(t) W(t)(M_"max" - M_b) - lambda_M M_b + epsilon_b(t) $

Trace-level recall support is:

$ R_mu(t) = sum_b a_(mu b) A_b(t) x_b(t) $

Memory linking is:

$ L_(mu nu)(t) = sum_b a_(mu b) a_(nu b) M_b(t) $

These equations are phenomenological. They define an executable bridge between the
biological motivation above and the simulator outputs analyzed below. The model
architecture is summarized in @fig-model-schematic.

// ─── Figures ─────────────────────────────────────────────────────────────────

#figure(
  image("figures2/fig1_slow-branch-level-accessability-as-a-structural-constraint-on-memory-linking.png", width: 100%),
  caption: [
    *Slow branch-level accessibility as a structural constraint on memory linking.*
    Fast dendritic access opens branches moment by moment, while replay-dependent slow
    structural writing changes which branches remain easier to recruit later. Two traces
    that share an overlap branch can become linked when replay consolidates that shared
    branch. Perturbing the overlap branch selectively harms linking, and targeted rescue
    restores linking only when the slow structural-writing pathway is available. The model
    tests branch-level accessibility, not direct molecular memory storage.
    *(A)* Biological rationale.
    *(B)* Minimal four-branch simulator topology.
    *(C)* Phenomenological variable definitions and update equations.
    *(D)* Interpretation boundaries.
    Take-home: the model separates fast access from a replay-written slow structural bias.
  ],
) <fig-model-schematic>

// ─── Claim Map ───────────────────────────────────────────────────────────────
= Claim Map and Evidence Ladder

#figure(
  kind: "table",
  supplement: "Table",
  caption: [Claim map for the simulator-first article. The slow-accessibility account is evaluated across mechanistic and output-level prediction families.],
  clean-table(
    columns: (1.2fr, 1.3fr, 1.6fr, 1.7fr, 1.5fr),
    header: ([*Claim*], [*Why it matters*], [*Test*], [*Result*], [*Interpretive limit*]),
    [Replay-linked selective reuse],
    [Recent activity should bias later reuse of overlapping branches beyond fixed allocation.],
    [Overlap-branch ($M_b$) gain and linking increase.],
    [The full model shows overlap writing and linking gain.],
    [Simulator result, not biological proof.],
    [Fast-vs-slow separation],
    [Context separation may arise from fast gating alone; durable linking requires slow writing.],
    [SIG-C vs SIG-A, SIG-B, SIG-E.],
    [SIG-C is non-diagnostic; discriminating result is the joint profile.],
    [No single signature is sufficient.],
    [Perturbation-sensitive linking],
    [Shared-route perturbation should harm linking more than single-trace recall.],
    [Focal overlap damage and targeted rescue.],
    [Linking drops after overlap damage and recovers most strongly after targeted rescue.],
    [Damage sensitivity alone is not unique to slow writing.],
    [Comparator discrimination],
    [Simpler mechanisms should fail at least part of the joint profile.],
    [Baseline and hard comparator matrices.],
    [No tested comparator reproduces the full structural-accessibility and rescue profile.],
    [Alternatives remain possible outside the tested set.],
    [Robustness],
    [The result should occupy a bounded regime rather than a single tuned point.],
    [One-at-a-time sweeps and two-parameter heatmaps.],
    [A coherent write--replay regime is identified.],
    [Robustness is bounded, not universal.],
    [Motif generalization],
    [The mechanism should survive larger allocation spaces but reveal topology limits.],
    [4--32 branches; weak, strong, chain, hub, and sparse motifs.],
    [Passes non-hub motifs; weak overlap and hub topology define boundaries.],
    [Motif generalization, not realistic dendritic-tree scaling.],
  ),
) <tab-claim-map>

The six claims tested in this paper, their tests, results, and interpretive limits are
listed in @tab-claim-map. The paper's overall evidence ladder and explicit claim
boundary are summarized in Supplementary Figure S5.5 (S5 Appendix).

// ─── Results ─────────────────────────────────────────────────────────────────
= Results

== 1. A traceable simulator makes the slow-writing hypothesis executable

We first instrumented the canonical branch-resolved simulator so that branch-level and
trace-level dynamics could be inspected directly. The canonical model contains four
branches, two traces, and one overlap branch shared by both traces. Encoding generates
branch-specific eligibility, replay generates consolidation support, and slow structural
writing updates $M_b$.

The trace export records fast activity ($x_b$), fast access, slow access, effective
access, eligibility ($E_b$), consolidation support ($P_b$), structural accessibility
($M_b$), recall support ($R_mu$), and linking ($L_(mu nu)$) across initialization,
encoding, pre-consolidation probe, replay consolidation, post-consolidation probe,
overlap damage, post-damage probe, targeted rescue, and post-rescue probe.

In the canonical run, the overlap branch showed selective slow structural gain during
replay consolidation. The linking score increased after consolidation, decreased after
overlap damage, and recovered after targeted rescue (@fig-canonical-traces). The
simulator therefore no longer depends on summary tables alone; the internal dynamics
are directly inspectable.

#figure(
  image("figures2/Fig_e023_02_canonical_traces.png", width: 100%),
  caption: [
    *Canonical simulator trace export.*
    Overlap branch (solid) and private branch (dashed) dynamics across initialization,
    encoding, consolidation, damage, and targeted rescue. Top: structural accessibility
    ($M_b$). Second: eligibility ($E_b$). Third: consolidation support ($P_b$). Bottom:
    linking score ($L$). Shading indicates simulation phase.
    Take-home: the full model exposes reproducible branch-level traces and a selective
    overlap-write, damage, and rescue cycle.
  ],
) <fig-canonical-traces>

These traces establish that the model's internal branch-level dynamics are inspectable
and reproducible. The result validates the simulator implementation and makes the
slow-writing hypothesis executable within the model, but it does not constitute
biological validation.

== 2. Replay-dependent slow writing produces a joint canonical signature profile

The canonical model was evaluated using five protected signatures:

- *SIG-A:* overlap-branch structural writing,
- *SIG-B:* linking gain after consolidation,
- *SIG-C:* context separation,
- *SIG-D:* linking-vs-recall dissociation under overlap damage,
- *SIG-E:* targeted rescue selectivity.

The full model passed all five signatures under the locked canonical protocol.
Overlap-branch structural writing was positive, linking increased after consolidation,
context separation was present, linking was more vulnerable than single-trace recall
under overlap damage, and targeted rescue produced strong normalized recovery relative to
generic no-precue consolidation.

However, not all signatures were equally diagnostic. Subsequent analyses showed that
SIG-C is largely architectural and context/allocation-related, while SIG-D can arise
from overlap-branch perturbation geometry. The diagnostic claim therefore rests on the
joint profile, especially the combination of slow structural writing, linking gain, and
rescue selectivity.

The perturbation-and-rescue logic behind these canonical signatures is illustrated in
Supplementary Figure S5.6 (S5 Appendix).

#figure(
  kind: "table",
  supplement: "Table",
  caption: [Canonical signature profile under the locked reference protocol. Take-home: the full model passes all five protected signatures, but SIG-C and SIG-D are interpretively weaker than the overlap-writing and rescue signatures.],
  clean-table(
    columns: (1fr, 1fr, 1fr, 2fr),
    header: ([*Signature*], [*Value*], [*Threshold*], [*Interpretation*]),
    [SIG-A], [0.1281], [> 0.02], [Selective overlap-branch structural strengthening.],
    [SIG-B], [0.2397], [> 0.05], [Linking increases after consolidation.],
    [SIG-C], [0.1545], [> 0.05], [Context separation is present but not model-specific.],
    [SIG-D], [20.98 pp], [> 5 pp], [Linking is much more damage-sensitive than single-trace recall.],
    [SIG-E], [1.57 (normalized)], [> 0.10], [Targeted rescue produces normalized recovery advantage of 1.57 over non-targeted baseline; gSIG-E convention (E019R/E022 protocol).],
  ),
) <tab-canonical-signatures>

Taken together, the canonical run shows a joint profile of overlap strengthening,
linking gain, perturbation-sensitive loss, and rescue selectivity (@tab-canonical-signatures).
The interpretive limit is that SIG-C and SIG-D are not specific markers on their own;
the main claim rests on the combined profile.

== 3. Simpler baseline comparators fail the joint profile

We next compared the full model against baseline comparators:

- *fast-context-only,*
- *replay-without-structure,*
- *random slow drift,*
- *fixed allocation only.*

All comparators preserved at least some partial behavior. In particular, context
separation passed broadly, confirming that SIG-C reflects fast gating or allocation
structure rather than slow writing. Several comparators also showed linking-vs-recall
dissociation under overlap damage, confirming that SIG-D is not diagnostic alone.

However, no simpler baseline reproduced the full SIG-A to SIG-E profile. Models without
structural writing failed overlap-branch ($M_b$) gain and linking growth. Random slow
drift failed specificity. Fixed allocation preserved overlap geometry but failed
replay-dependent dynamic updating.

Under canonical parameters, no tested simple baseline reproduced the full joint
signature profile (@fig-comparator-matrix; @tab-comparator-summary). This narrows the
mechanism class under study, but it does not rule out every conceivable alternative
outside the tested comparator set.

== 4. Hard comparators reproduce isolated output-level effects but not the full structural-rescue profile

To make the comparator test fairer, we added harder alternatives:

- *Hebbian weight-only,*
- *soma/global-gain-only,*
- *shuffled replay,*
- *eligibility-only,*
- *resource-only.*

These comparators were evaluated at both structural and output-level metrics. This
distinction is important because a weight-only comparator should not be dismissed merely
because it lacks $M_b$. The relevant question is whether it reproduces the same
output-level profile and the same structural-accessibility/rescue profile.

Hebbian weight-only produced strong output-level linking but no branch-specific structural
writing and no rescue of a written structural state. Eligibility-only and resource-only
comparators produced transient effects but no persistent recoverable structure. Soma/global
gain produced broad activation without branch-specific structural writing. Shuffled replay
produced a small apparent overlap advantage in the smallest four-branch motif but failed
in larger allocation spaces.

Across canonical, strong-overlap, chain-overlap, and sparse-random motifs, no hard
comparator reproduced the full structural-accessibility and rescue profile of the full
model.

The logic of this fair two-level comparator test is summarized in
Supplementary Figure S5.7 (S5 Appendix).

#figure(
  image("figures2/Fig_e023_03_comparator_matrix.png", width: 100%),
  caption: [
    *Comparator discrimination matrix.*
    Left: structural signature pass/fail for SIG-A through SIG-E. Right: output-level
    metrics (B1 linking gain, B3 damage sensitivity, B5 recovery index). Top
    section: baseline comparators (E018). Bottom section: hard comparators (E022). Green
    indicates PASS, and red indicates fail. Dashed line separates baseline from hard comparator sections.
    Take-home: several alternatives reproduce partial output-level effects, but none
    reproduce the full structural-accessibility and rescue profile.
  ],
) <fig-comparator-matrix>

#figure(
  kind: "table",
  supplement: "Table",
  caption: [Comparator summary across baseline and hard alternatives. Take-home: some alternatives reproduce partial output-level metrics, but none match the full structural-accessibility and rescue profile.],
  clean-table(
    columns: (1.3fr, 1.7fr, 1.3fr, 1.7fr, 0.8fr),
    header: ([*Comparator*], [*Mechanism removed or replaced*], [*Structural summary*], [*Output-level summary*], [*Joint verdict*]),
    [full_model], [All mechanisms active.], [SIG-A = 0.128; SIG-B = 0.240; gSIG-E = 1.57.], [B1 = 0.239; B3 = 0.141; B5 = 1.57.], [Pass],
    [fast_context_only], [Fast gating retained; $M_b$ frozen.], [SIG-A/B/E = 0; only SIG-C and SIG-D pass.], [Context separation and damage sensitivity remain architecture-driven.], [Fail],
    [replay_no_structure], [Replay without persistent write.], [SIG-B = -0.018; SIG-E = 0.], [Transient replay is insufficient for durable linking.], [Fail],
    [random_slow_drift], [Non-specific slow drift with matched scale.], [SIG-A = -0.089; SIG-B = -0.065.], [Produces non-specific or unstable effects.], [Fail],
    [fixed_allocation_only], [Static overlap without dynamic update.], [SIG-A/B/E = 0.], [Preserves context geometry but no replay-written gain.], [Fail],
    [hebbian_weight_only], [Weight updates only.], [gSIG-A = 0 across tested motifs.], [Canonical B1 = 4.18, but B5 = 0.], [Fail],
    [soma_global_gain_only], [Branch-independent gain.], [gSIG-A = 0; chain gSIG-B = 0.140.], [Canonical B1 = 0.198; B5 = 1.88 without specificity.], [Fail],
    [eligibility_only], [Transient eligibility without retained structure.], [gSIG-A = 0.], [Canonical B1 = 0.398; B5 = 0.], [Fail],
    [resource_only], [Capture resources without retained $M_b$.], [gSIG-A = 0.], [Canonical B1 = 0.406; B5 = 0.], [Fail],
    [shuffled_replay], [Replay identity scrambled.], [gSIG-A = 0.091 at $n$=4 but decays with scale.], [Canonical B1 = 0.191; B5 = 1.82, but specificity collapses.], [Fail],
  ),
) <tab-comparator-summary>

The fair comparator question is therefore two-level: whether an alternative reproduces
the same output-level linking and recall pattern, and whether it reproduces the proposed
structural-accessibility mechanism. On that basis, weight-only and gain-based mechanisms
remain serious partial alternatives for some outputs, but none matched the full
structural, perturbation, rescue, and specificity profile (see @fig-comparator-matrix
and @tab-comparator-summary for full per-comparator values).

== 5. One-at-a-time robustness identifies bounded functional ranges

We next varied key parameters one at a time while preserving the locked signature
definitions. The model survived meaningful variation in structural learning rate, replay
gain, eligibility decay, structural decay, structural noise, context gain, timing gap,
overlap strength, and readout threshold.

The result was not uniform success. Failure boundaries were informative. Structural
learning rate near zero failed slow writing and linking. Replay gain near zero failed
replay-dependent consolidation. Very high structural decay eroded persistence. Long
timing gaps combined with eligibility decay impaired consolidation. Weak overlap below
approximately 0.40 failed linking. Moderate structural noise was tolerated, but high
noise combined with low replay produced instability.

Context gain had little effect on the full profile because SIG-C was already supported
by allocation geometry. This reinforced the interpretation that context separation is an
auxiliary fast-access signature rather than a diagnostic slow-writing signature.

One-at-a-time robustness therefore shows that the full model's joint profile is not a
single-run artifact; it survives meaningful parameter variation within bounded regimes
(@fig-robustness-landscape; @tab-robustness-summary). This does not imply robustness
to all parameter interactions, which is why the paired heatmaps below remain necessary.

== 6. Two-parameter heatmaps reveal a write--replay regime and temporal eligibility boundary

We then tested biologically meaningful parameter interactions. Six parameter pairs were
evaluated:

- structural learning rate × replay gain,
- eligibility decay × timing gap,
- overlap strength × replay gain,
- structural decay × structural learning rate,
- structural noise × replay gain,
- context gain × structural learning rate.

The structural learning rate × replay gain heatmap showed a coherent write--replay
regime. Both parameters were required: low structural learning rate or low replay gain
caused failure. The eligibility decay × timing gap heatmap showed a temporal boundary:
long delays combined with fast eligibility decay exhausted the tag before consolidation.
The overlap strength × replay gain heatmap showed that linking requires sufficient
overlap and sufficient replay. Structural decay imposed a persistence limit even when
learning rate was high. Structural noise was broadly tolerated except at extreme noise
and low replay. Context gain again confirmed that context separation is separable from
slow structural writing.

#figure(
  image("figures2/Fig_e023_04_robustness_landscape.png", width: 100%),
  caption: [
    *Robustness landscape.*
    *(A)* One-at-a-time parameter sweep: % of values for which all five signatures pass
    jointly (green ≥75%, orange 50--74%).
    *(B)* Two-parameter heatmap, structural_lr × replay_gain (star = canonical).
    *(C)* overlap_strength × replay_gain. Green indicates joint pass, and red indicates fail.
    Take-home: the mechanism occupies a bounded functional regime rather than a single
    tuned parameter point.
  ],
) <fig-robustness-landscape>

#figure(
  kind: "table",
  supplement: "Table",
  caption: [Robustness summary across one-at-a-time sweeps and two-parameter heatmaps. Take-home: robustness is broad but bounded by write strength, replay strength, timing, and overlap.],
  clean-table(
    columns: (1.2fr, 1.8fr, 1.5fr, 1.6fr),
    header: ([*Analysis*], [*Quantitative summary*], [*Main boundary*], [*Interpretation*]),
    [One-at-a-time sweeps], [6 of 9 parameters showed at least 75% joint pass; context_gain, eligibility_decay, and readout_threshold were 100%.], [Overlap strength (62.5%) and structural noise (57.1%) define the narrowest tested ranges.], [The profile is not a single-run artifact.],
    [structural_lr × replay_gain], [70 of 99 cells passed (70.7%).], [Low write rate or low replay collapses SIG-A, SIG-B, and SIG-E.], [Both persistent writing and replay are required.],
    [eligibility_decay × timing_gap], [45 of 72 cells passed (62.5%).], [Long delays with rapid eligibility decay fail.], [Tag lifetime limits consolidation timing.],
    [overlap_strength × replay_gain], [37 of 72 cells passed (51.4%).], [Overlap below about 0.40 fails even with replay.], [Sufficient but not universal overlap is required.],
    [structural_noise × replay_gain], [55 of 63 cells passed (87.3%).], [Extreme noise with weak replay destabilizes the profile.], [Moderate noise tolerance is broad but not unlimited.],
  ),
) <tab-robustness-summary>

The robustness analyses therefore support a bounded functional regime rather than a
single tuned point (@fig-robustness-landscape, panel B–C; @tab-robustness-summary).
The interpretive limit is that these heatmaps cover selected parameter pairs, not
arbitrary high-dimensional combinations.

== 7. Scaling and motif tests show generalization with topology-dependent specificity limits

The canonical four-branch motif is useful for exposition, but it could be dismissed as a
hand-built toy model. We therefore generalized the simulator to 4, 8, 16, and 32
branches and tested multiple motif classes:

- *canonical,*
- *weak overlap,*
- *strong overlap,*
- *chain overlap,*
- *hub overlap,*
- *sparse random allocation.*

The mechanism generalized beyond four branches in canonical and strong-overlap motifs.
Chain-overlap motifs showed local linking stronger than distant linking, with modest
leakage. Sparse-random motifs also passed under tested densities and seeds, although
density could create universal linking at larger scales.

Weak-overlap motifs failed as expected, confirming the overlap threshold identified in
robustness tests. Hub-overlap motifs produced strong structural writing and linking but
failed specificity: because all traces shared the same hub branch, all trace pairs became
linked. This is not a clean success. It is an over-linking boundary condition.

#figure(
  image("figures2/Fig_e023_05_motif_scaling.png", width: 100%),
  caption: [
    *Scaling and motif generalization.*
    *(A)* Mean gSIG-A (structural write advantage) for six motif types.
    *(B)* gSIG-A and gSIG-E (rescue) as branch count increases for the canonical motif
    (4--32 branches).
    *(C)* False-linking rate for multi-trace motifs (chain, hub, sparse); dashed lines
    mark thresholds (good below 0.25, moderate below 0.50).
    Take-home: the mechanism generalizes across tested branch counts and motif classes
    only within overlap-topology boundaries that preserve specificity.
  ],
) <fig-motif-scaling>

#figure(
  kind: "table",
  supplement: "Table",
  caption: [Motif and specificity summary. Take-home: selected non-hub motifs preserve the mechanism, whereas weak overlap and hub topology define clear boundary conditions.],
  clean-table(
    columns: (1.1fr, 1fr, 1.6fr, 1.8fr, 1.3fr),
    header: ([*Motif*], [*Joint pass*], [*Key values*], [*Specificity interpretation*], [*Limit*]),
    [canonical], [4/4 scales], [mean gSIG-A = 0.234; mean gSIG-B = 0.239; false-linking = 0.00], [Reference behavior is stable from 4 to 32 branches.], [Scaling of neutral branches only.],
    [weak_overlap], [0/4 scales], [mean gSIG-A = -0.074; mean gSIG-B = -0.040], [Expected failure below the overlap threshold.], [Boundary, not defect.],
    [strong_overlap], [4/4 scales], [mean gSIG-B = 0.295; mean gSIG-E = 1.566], [Specific linking success with stronger overlap.], [Two-trace motif only.],
    [chain_overlap], [3/3 scales], [mean gSIG-A = 0.217; false-linking ≈ 0.192], [Local linking is preserved with modest leakage.], [Partial specificity only.],
    [hub_overlap], [mechanistic pass / specificity fail], [mean gSIG-A = 0.243; mean gSIG-B = 0.234], [Universal shared overlap produces over-linking.], [Boundary condition, not clean success.],
    [sparse_random], [6/6 runs], [mean gSIG-A = 0.160; false-linking ≈ 0.231], [Partial specificity survives under tested densities and seeds.], [Density-dependent regime.],
  ),
) <tab-motif-summary>

The mechanism therefore generalizes across tested branch counts and motif classes in
selected non-hub topologies, but specificity depends on overlap structure
(@fig-motif-scaling; @tab-motif-summary). This is motif generalization, not realistic
dendritic-tree scaling.

== 8. Shuffled replay fails as branch allocation space expands

The hard-comparator analysis showed a small apparent structural signal for shuffled
replay in the smallest four-branch motifs. We therefore audited shuffled replay across
branch counts and seeds.

With 20 seeds, the apparent effect was shown to be a high-end small-sample draw rather
than a stable signal. In the four-branch motif, shuffled replay reached only about 7% of
the full-model overlap advantage on average. At 8 branches the ratio fell below 5%, and
by 16--32 branches it approached zero. Variance also decreased with increasing branch
count.

This result clarifies the role of replay identity. Replay alone is not sufficient.
Identity-preserving replay is required for scalable structural specificity.

#figure(
  image("figures2/Fig_e023_06_shuffled_replay_audit.png", width: 100%),
  caption: [
    *Shuffled replay scaling audit (E022R).*
    *(A)* gSIG-A for full model (dashed) and shuffled replay mean ± SD (band) across
    4--32 branches.
    *(B)* Ratio of shuffled mean to full-model gSIG-A; ratio falls from ~7% ($n$=4) to
    below 0.1% ($n$=32).
    *(C)* Distribution of shuffled gSIG-A across 20 seeds at $n$=4 (boxes), with
    full-model reference (green dashes). The apparent small-motif match is a sampling
    artefact that decays with branch allocation space.
    Take-home: replay alone is insufficient; scalable specificity requires
    identity-preserving replay.
  ],
) <fig-shuffled-replay>

Shuffled replay can weakly mimic overlap writing in the smallest allocation space by
chance, but this effect decays rapidly with branch count (@fig-shuffled-replay). The
result concerns the implemented replay-identity disruption tested here, not every
possible replay-disruption model.

// ─── Discussion ──────────────────────────────────────────────────────────────
= Discussion

== What the simulator supports

The simulator supports a bounded, executable claim: replay-dependent slow
branch-level structural writing can reproduce a joint profile of memory-linking
signatures that tested alternatives do not reproduce. The full model produces traceable
overlap-branch strengthening, replay-linked linking gain, perturbation-sensitive loss,
and targeted rescue. The result survives one-at-a-time and two-parameter robustness
tests, generalizes across tested branch counts and motif classes, and shows meaningful
topology-dependent boundaries.

The strongest contribution is not any single signature. Context separation is not
diagnostic of slow writing. Perturbation-sensitive linking is not diagnostic alone.
The contribution is the joint profile: structural writing, linking gain, perturbation
response, rescue selectivity, robustness, and specificity across non-hub motifs.

== What the simulator does not support

These results do not establish that cytoskeletal structures directly encode memory. $M_b$
is a phenomenological latent variable. Direct biological validation of slow structural
accessibility ($M_b$) is not supported here. Open-data analyses are supplementary and
exploratory.

== Biological interpretation

The biological interpretation remains plural and cautious. $M_b$ may represent a family
of slow branch-local constraints: spine geometry, actin remodeling, microtubule
invasion, local translation, transport readiness, mitochondrial or metabolic support, or
other structural and biochemical processes. The key idea is not that one molecule stores
memory, but that branch-local structural state can bias future access.

This framing fits a broader view of dendrites as active, history-sensitive access
structures. Hippocampal memory linking provides the current testbed
@Choucry2024MemoryLinkingIdentity @Kastellakis2023DendriticEngram @Sehgal2025ContextLinking
@Guskjolen2023EngramNeurons @Zaki2024OfflineLinking, but similar fast/slow access
separations may be relevant in prefrontal contextual control
@WangYang2018Routing @Keller2020ContextualModulation @Bos2025GainModulation
@Olah2025HCNGating, working memory stabilization, and systems-level consolidation
@Frankland2005RecentRemoteMemories @Wang2024SleepDependentEngramReactivation
@TononiCirelli2020SleepPlasticity @ReyesResina2021SleepConsolidation. Those extensions
remain speculative until separately implemented and tested.

The shuffled-replay audit sharpens this biological interpretation. It shows that replay
occurrence alone is not enough; scalable specificity depends on replay preserving branch
identity. In biological terms, that constrains the relevant mechanism class toward
processes that can stabilize or revisit previously eligible branch-local states rather
than merely increasing global post-encoding activity.

== Failure modes are informative

Several failures strengthen the model's interpretability. Weak overlap fails. Excessive
hub overlap produces over-linking. Long timing gaps fail when eligibility decays. High
structural decay erodes persistence. Shuffled replay fails as allocation space expands.
These failures show that the model is not a free-floating explanation that succeeds
everywhere. It defines boundary conditions.

== Future tests

The most direct future biological tests would require longitudinal branch-resolved
recordings across linked memory formation, replay, perturbation, and rescue. Stronger
evidence would come from experiments that dissociate fast contextual gating from slower
replay-dependent stabilization; perturb overlap-relevant dendritic mechanisms; and test
whether linking is more vulnerable than single-trace recall under targeted structural
disruption @Sehgal2025ContextLinking @Kastellakis2023DendriticEngram
@Uytiepo2025EngramArchitecture.

Future simulator work should extend beyond branch-count scaling toward more biologically
structured dendritic trees, including proximal/distal organization, apical and basal
compartments, spine distributions, inhibitory gating, branch-specific thresholds, and
hippocampus--prefrontal differences.

// ─── Materials and Methods ───────────────────────────────────────────────────
= Materials and Methods

== Overview

The revised study is organized as a simulator-first evidence ladder. The primary
experiments test whether replay-dependent slow branch-level structural writing is needed
to reproduce a joint signature profile in memory linking. Open-data bridge analyses are
not part of the main evidence ladder and are described in S3 Appendix.

== Study design and evidence tiers

The study contains three explicit evidence tiers. The primary evidence tier contains the
simulator experiments `E017--E022R` together with the assembled article figures,
as listed in @tab-experiment-registry. A second legacy tier contains earlier experiment
families retained for lineage and historical context but not used as the main evidence
ladder for the current manuscript. A third supplementary tier contains exploratory
open-data bridge analyses, reported in S3 Appendix, that test downstream observable
consequences without directly measuring $M_b$.

#figure(
  kind: "table",
  supplement: "Table",
  caption: [Experiment registry for the simulator-first manuscript.],
  clean-table(
    columns: (0.9fr, 2.4fr, 1.2fr),
    header: ([*Experiment*], [*Purpose*], [*Primary output*]),
    [E017], [Trace export and canonical signature profile], [Fig. 2; Table 2],
    [E018], [Baseline comparator matrix], [Fig. 3; Table 3],
    [E019], [One-at-a-time robustness], [Fig. 4; Table 4],
    [E020], [Two-parameter heatmaps], [Fig. 4; Table 4],
    [E021], [Scaling and motif generalization], [Fig. 5; Table 5],
    [E021R], [Specificity gate and motif language], [Fig. 5 interpretation],
    [E022], [Hard comparators], [Fig. 3; Table 3],
    [E022R], [Shuffled replay audit], [Fig. 6],
  ),
) <tab-experiment-registry>

Detailed reproduction routing, script-level mapping, and legacy experiment lineage are
provided in `reviewer_slow_branch_level_accessibility.ipynb` and
`CLAIMS_TO_EXPERIMENTS.md`. Older conceptual assets are retained there as supplementary framing material rather
than being expanded inside this manuscript table.

== Simulator implementation and experiment families

The simulator is implemented in Python in `src/cytodend_accessmodel/` and is exposed
through experiment scripts in `experiments/`. The canonical traceable reference stack is
`E017` plus `E018`, which together define the locked four-branch/two-trace protocol,
export branch-level traces, and evaluate the baseline comparator set. `E019` and `E020`
extend that reference stack with one-at-a-time and paired-parameter robustness analyses.
`E021` and `E021R` extend the model to larger branch counts and overlap motifs. `E022`
and `E022R` add hard comparators and the shuffled-replay specificity audit. Full
computational routing, software/resource identifiers, and reproducibility details are
collected in S4 Appendix.

== Simulator architecture

The canonical simulator contains four branches and two traces. One branch is shared
between the traces and serves as the overlap branch. Each trace also has private
branches. Cue input drives trace-specific branch activation, eligibility marks recently
active branches, replay generates consolidation support, and slow structural writing
updates $M_b$. Recall support and linking are computed from trace-to-branch allocations
and branch accessibility states.

== Trace export

The simulator exports branch-level traces, trace-support traces, and linking traces.
Exported branch variables include phase, branch ID, fast activity, fast access, slow
access, effective access, eligibility, consolidation support, structural accessibility,
input drive, and replay drive. Trace-level outputs include recall support. Pair-level
outputs include linking scores and branch contributions.

== Replication, determinism, and stochastic runs

The canonical `E017` and `E018` reference runs are deterministic because
`structural_noise = 0.0` and the protocol is fixed in code. Stochastic replication is
introduced only in experiment families that explicitly audit noise or replay shuffling.
In `E019`, the structural-noise sweep uses `10` predeclared seeds. In `E020`, the
`structural_noise × replay_gain` heatmap likewise uses `10` predeclared seeds per noisy
grid cell. In `E022R`, shuffled replay is evaluated with `20` shuffled-replay seeds per
condition. These stochastic runs are technical robustness replicates of the implemented
simulator rather than biological replicates.

== Protected signatures

The canonical signatures are:

- *SIG-A:* overlap-branch structural writing,
- *SIG-B:* linking gain after consolidation,
- *SIG-C:* context separation,
- *SIG-D:* linking-vs-recall dissociation under overlap damage,
- *SIG-E:* targeted rescue selectivity.

SIG-E is reported as normalized recovery difference, not as percentage points.

== Comparator models

Baseline comparators included fast-context-only, replay-without-structure, random slow
drift, and fixed allocation only. Hard comparators included Hebbian weight-only,
soma/global-gain-only, shuffled replay, eligibility-only, and resource-only variants.
Comparators were evaluated with both structural signatures and output-level metrics
so that alternatives lacking $M_b$ were not unfairly dismissed.

== Robustness analyses

One-at-a-time sweeps varied structural learning rate, replay gain, eligibility decay,
structural decay, structural noise, context gain, timing gap, overlap strength, and
readout threshold. Two-parameter heatmaps tested interactions among structural learning
rate, replay gain, eligibility decay, timing gap, overlap strength, structural decay,
structural noise, and context gain.

== Motif and scaling analyses

A generalized motif engine generated canonical, weak-overlap, strong-overlap,
chain-overlap, hub-overlap, and sparse-random allocation motifs. Branch counts ranged
from 4 to 32. Multi-trace motifs included chain, hub, and sparse-random structures.
Generalized signatures measured expected-overlap structural writing, expected-pair
linking gain, perturbation specificity, rescue selectivity, and false-linking or
specificity where defined.

== Shuffled replay audit

Shuffled replay was audited across branch counts and seeds to determine whether
small-motif apparent structural matching was stable. Full-model reference values were
compared against shuffled-replay means, standard deviations, and ratios across 4, 8, 16,
and 32 branches with 20 seeds per condition.

== Statistical and computational reporting conventions

Analyses were designed for Python `>=3.10`, and project dependencies are declared in
`pyproject.toml`. Core software used in the simulator and supplementary open-data
workflows includes `numpy`, `pyyaml`, `torch`, `dandi`, `pynwb`, and `matplotlib`.
Software identifiers and dataset identifiers are listed in S4 Appendix.

The primary simulator claims are based on deterministic reference runs, explicit
signature thresholds, and predeclared comparator definitions rather than on fitting a
model to maximize one summary statistic. Technical replication was used only in the
stochastic robustness and shuffled-replay audits, where seed lists were predeclared in
the corresponding experiment scripts.

Investigators were not blinded to model condition because all simulator outputs were
generated and labeled by automated scripts with predefined configuration names.
Experimental subjects were not randomized into groups because the primary evidence ladder
is a computational simulator study and does not assign biological subjects to
intervention groups. We did not check for sample sizes using a power analysis because the
primary simulator evidence ladder is based on deterministic reference runs plus
predeclared robustness audits rather than on between-subject or within-subject biological
sampling. No simulator runs in the primary evidence ladder were excluded on the basis of
their outcomes.

No new human participants were recruited for this study. No live-animal experiments were
performed by the author for the simulator study. No cell lines, antibodies, or collected
specimens were used in the primary simulator evidence ladder. Supplementary open-data
analyses reuse public third-party DANDI datasets under their original archive records.

== Open-data bridge analyses

These analyses are retained as supplementary exploratory bridge analyses and are not
used to support the primary simulator-first claims.
Open-data analyses using DANDI datasets are reported only in S3 Appendix. They do not
directly measure $M_b$, dendritic branch accessibility, cytoskeletal writing, or a
molecular memory code.

// ─── Data and Code Availability ──────────────────────────────────────────────
= Data and Code Availability

Code availability: all author-generated code, analysis scripts, configuration files,
figures, and manuscript-source material associated with this study are available in the
public `cytodendaccessmodel` GitHub repository
(`https://github.com/NoeticDiffusion/cytodendaccessmodel`) and in the
manuscript-matched Zenodo snapshot (DOI `10.5281/zenodo.20813821`;
`https://doi.org/10.5281/zenodo.20813821`). The code is released under the GNU General
Public License v3.0 (`GPL-3.0`).

Data availability: primary simulator outputs supporting the main evidence ladder are
organized under `results/e017_traceable_simulator_core`,
`results/e018_comparator_trace_matrix`, `results/e019_one_at_a_time_parameter_robustness`,
`results/e020_two_parameter_robustness_heatmaps`,
`results/e021_scaling_and_motif_generalization`,
`results/e021r_generalized_specificity_gate`, `results/e022_hard_comparators`, and
`results/e022r_shuffled_replay_scaling_audit`. Reproducibility routing is provided in
`reviewer_slow_branch_level_accessibility.ipynb` and `CLAIMS_TO_EXPERIMENTS.md`.

Data availability: empirical datasets used in the supplementary open-data bridge analyses
are not redistributed by the author. They remain accessible through their original DANDI
Archive records (`000718`, `000336`, `001710`), while derived manuscript-facing outputs
are organized under `data/dandi/triage/000718`, `data/dandi/triage/000336`, and
`data/dandi/triage/001710`.

// ─── Acknowledgments ─────────────────────────────────────────────────────────
= Acknowledgments

The author gratefully acknowledges the investigators who generated and shared the public
datasets discussed in the supplementary open-data bridge analyses.

The author also acknowledges editorial, analytical, and software-organization assistance
from large language models during literature synthesis, structured critique, drafting,
and revision. These tools are not authors and did not provide peer review. The author
reviewed and approved all scientific claims, code, analyses, citations, and final
wording, and takes full responsibility for the manuscript.

// ─── Bibliography ────────────────────────────────────────────────────────────
#bibliography("references_slow_branch_level_accessibility.bib")

// ─── Supporting Information ──────────────────────────────────────────────────

#pagebreak()

= Supporting Information

#supplement-box([S1 Appendix. Mathematical framework and theory-to-executable mapping])[
  Extended formalization of the branch-accessibility model, including fast/slow access
  factorization, eligibility dynamics, consolidation support, slow structural writing,
  recall support, linking metrics, and the mapping between biological notation and
  simulator variables. This appendix also documents the interpretation of $M_b$ as a
  phenomenological slow accessibility variable rather than a direct molecular memory
  substrate.
]

#supplement-box([S2 Appendix. Simulator architecture, robustness, comparators, and motif tests])[
  Detailed simulator architecture, canonical parameter set, trace-export schema,
  protected signature definitions, comparator baseline definitions, hard comparator
  implementations, one-at-a-time robustness, two-parameter heatmaps, motif generator,
  scaling analyses, specificity metrics, and shuffled replay audit. This appendix
  contains reproducibility details for E017--E022R at the simulator-design level.
]

#supplement-box([S3 Appendix. Exploratory open-data bridge analyses])[
  Exploratory DANDI bridge analyses moved from the main article. These analyses test
  downstream observable consequences of the branch-accessibility hypothesis, including
  offline reactivation, structured inter-plane coupling, and perturbation-sensitive
  cross-day stabilization. They do not directly observe $M_b$, dendritic branch
  accessibility, cytoskeletal memory writing, or molecular memory storage. Their role is
  to suggest empirical contact points and constraints for future work, not to validate
  the simulator.
]

#supplement-box([S4 Appendix. Computational materials, software resources, and reporting conventions])[
  Computational reporting supplement for the simulator-first manuscript, including
  software and dataset identifiers, software/version routing, detailed experiment-family
  design notes, stochastic-replication policy, inclusion/exclusion conventions, and
  reproducibility entry points for the code and output stack.
]

#supplement-box([S5 Appendix. Biological motivation figures and graphical schematics])[
  Graphical schematics and conceptual figures that accompany the biological framing of
  the simulator-first manuscript. These figures are supplemental because the main article
  is simulator-first; they are retained here as orientation material. Contents: S5.1 —
  biological motivation overview (microtubule/spine-neck gating and accessibility
  concept); S5.2 — structural impedance gating at the spine neck; S5.3 — minimal
  biological formalization; S5.4 — memory linking and contextual retrieval via shared
  branches; S5.5 — evidence ladder and claim boundary; S5.6 — overlap damage and
  targeted rescue logic; S5.7 — comparator fairness and model-discrimination logic.
]
