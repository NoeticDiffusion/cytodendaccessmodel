// Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking
// Typst source — simulator-first manuscript draft, updated after E025.
// The local template provides layout conventions; journal formatting remains to be finalized.

#import "template.typ": essay-template, abstract-block, supplement-box, note-line, clean-table, wide-table

#show: essay-template.with(
  title: [Testing Slow Branch-Level Accessibility in a Simulator of Memory Linking],
  short_title: [Slow branch access in memory linking],
  authors: [Robin Langell],
  affiliations: [Independent Researcher],
  correspondence: [hello\@noeticdiffusion.com],
  contributor_notes: [
    #align(center)[
      #text(size: 10pt, style: "italic")[
        A tuned, held-out test of a phenomenological slow branch-access hypothesis
      ]
    ]
  ],
  abstract: [
    We test a branch-resolved simulator in which a phenomenological slow accessibility state, $M_b$, changes later branch recruitment after eligibility- and replay-dependent updates. E025 compared ten simulator families under a shared probe and readout, with equal-budget tuning and a locked holdout of 100 topology draws. The tuned full model's mean shared-branch-specific change was $D=0.0227$ (95% bootstrap CI 0.02254--0.02286). Hebbian weight-only and resource-only models produced larger means ($D=0.0933$ and $D=0.0943$) and exceeded the full model under the preregistered, multiplicity-corrected comparison. The full model exceeded five alternatives and was practically equivalent to shuffled replay; eligibility-only yielded no qualifying candidate under the prespecified baseline and functional criteria. These simulator results do not support a unique output-level advantage or necessity for slow branch writing. They do not validate $M_b$ biologically or identify a molecular memory code.
  ],
)

// ─── Author Summary ──────────────────────────────────────────────────────────
#supplement-box([Author Summary])[
  Dendritic branches can carry local changes that affect later plasticity, and recent work
  has connected branch-local plasticity to memory linking. This paper asks a narrower
  computational question: does a distinct slow branch-access state improve a simulator's
  shared-probe prediction beyond matched weight-, resource-, and gating-based alternatives?
  We do not claim that dendrites or cytoskeletal structures directly store memory content.

  We compare ten simulator families using one common probe and readout, equal tuning
  budgets, and a held-out set of generated topologies. Two simpler families, Hebbian
  weight-only and resource-only, produced larger primary effects than the full model.
  The result is a bounded model-comparison finding, not evidence that slow writing is
  required or that a cytoskeletal process stores memory.
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
@Sehgal2025ContextLinking @Zaki2025Engram. Recent in vivo work reports reactivation and
structural plasticity in overlapping dendritic segments during linked-context learning,
and a network model in which local dendritic mechanisms contribute to co-recall
@Sehgal2025ContextLinking. We therefore do not claim that branch-local dendritic
plasticity or its role in memory linking is itself new. Those findings motivate the
narrower question tested here: does a distinct slow, replay-dependent branch-access
state add output-level explanatory value beyond matched alternatives?

The open question is whether a slow branch-level state adds a distinctive output-level
prediction beyond contextual routing and gain modulation @WangYang2018Routing
@Keller2020ContextualModulation @Bos2025GainModulation, fixed overlap geometry,
weight-based associative strengthening @Podlaski2025HighCapacity @Koch1999Biophysics
@Knierim2016Tracking, replay without persistent structural change, random slow drift, or
global gain. Earlier simulator experiments traced these mechanisms and their internal
signatures, but used comparator-specific scores. E024 introduced a common probe within
the branch-resolved simulator. E025 then gave each family the same tuning budget,
selected candidates without access to holdout outcomes, and compared them on new generated
topologies.

This paper contributes (i) a code-exact, branch-resolved formalization of a slow access
state distinct from fast cue gating, (ii) a preregistered equal-budget comparison against
nine alternative simulator families using a common output and independent holdout, and
(iii) a bounded result: after tuning, Hebbian weight-only and resource-only alternatives
exceeded the full model on the primary readout, while the full model exceeded five other
families and was practically equivalent to shuffled replay. The evidence concerns these
simulator implementations and task. It does not establish a unique biological mechanism
or a molecular or cytoskeletal memory code.

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
  Increased cross-trace facilitation. The current $L_(mu nu)$ score measures shared
  structural accessibility; E025 separately compares models with a common recall-based
  probe outcome.
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
  A readout defined with the same probe, timing, output equation, and scale for every
  model. E025 uses this shared readout with predeclared, model-defined state adapters and
  equal-budget tuning; its results compare the specified simulator implementations and
  do not make every internal state or mechanism identical across model families.
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
is narrower: replay-dependent slow writing is one candidate explanation for the joint
profile of overlap-branch structural gain, durable linking, targeted rescue, and
topology-dependent specificity. Whether it makes a distinct behavioral prediction
compared with matched alternatives remains the central model-comparison question;
structural measures alone cannot answer it.

== Minimal formalization

The primary experiments use the simulator's discrete updates below. All quantities are
dimensionless simulator values. A cue call and a consolidation pass are distinct state
updates; neither is calibrated to seconds, minutes, or hours. Here $I_b$ is the cue drive,
$c_b$ the branch context bias, $h_b$ local inhibition, and $q_b$ the mean spine-access
contribution (set to 1 when a branch has no spines).

At a cue call, fast and slow access and the resulting branch activation are:

$ A_b^f = sigma(g_f I_b + g_c c_b + g_"sp" q_b - g_i h_b), quad
  A_b^s = sigma(g_M M_b), quad
  A_b = op("clip")_(0,1)(A_b^f A_b^s), quad
  x_b = A_b I_b. $

The cue also updates the local tag and passively decays consolidation support:

$ E_b^"cue" = op("clip")_(0,1)((1-d_E) E_b + abs(x_b)), quad
  P_b^"cue" = op("clip")_(0,1)((1-d_P) P_b). $

There is no integrated activity state or recurrent/replay input in `apply_cue`: replay
selects traces and updates consolidation support in a separate consolidation call.
For replay set $S$ and trace replay priorities $pi_mu$, direct branch overlap is the
priority-weighted mean of allocation weights $a_(mu b)$ for $mu in S$, or zero when
there are no selected traces or the total priority is zero. Neighbor spillover, when
enabled, is added and clipped to $[0,1]$ to give $r_b$. Replay IDs are supplied by the
consolidation window or selected by its context rule. Specifically, a nonempty
`replay_trace_ids` list supplies the set; an empty list invokes the default selection of
all context-compatible traces.

At a consolidation pass, support is updated first; if `translation_budget` is positive
and total support exceeds it, all $P_b$ values are scaled by the same factor. The
structural state then changes according to:

$ P_b^+ = op("clip")_(0,1)((1-d_P) P_b + g_R r_b + g_"sleep" s r_b). $

$ M_b^+ = op("clip")_(0,M_"max")(M_b + eta E_b P_b^+ max(0,W) (1-M_b/M_"max")
  - lambda_b M_b + epsilon_b). $

$ E_b^+ = op("clip")_(0,1)((1-d_E) E_b). $

Here $P_b^+$ in the $M_b$ equation denotes the readiness value after optional global
budget normalization; the structural formula assumes $M_"max" > 0$.
Here $W$ is the window's global `modulatory_drive`, $s$ is its `sleep_drive`, and
$epsilon_b$ is one zero-mean Gaussian draw per branch and consolidation pass, with
standard deviation set by that branch's `noise_scale` (zero in deterministic runs).
Eligibility is used before it decays. The update is clipped to the branch-specific
interval $[0,M_"max"]$.

For a cue/readout, recall support and the structural linking diagnostic are:

$ R_mu = C_mu sum_b a_(mu b) x_b, quad
  L_(mu nu) = sum_b a_(mu b) a_(nu b) M_b. $

$C_mu$ is 1 unless both active and trace contexts are non-null and differ; in that case
it is $max(0,1-kappa)$, where $kappa$ is the context-mismatch penalty. Because $x_b$
already includes $A_b$, the readout does not
multiply by accessibility again. $L_(mu nu)$ is an internal, $M_b$-dependent structural
diagnostic; it is not by itself a model-independent behavioral measure of linking.
The simulator's `apply_cue` call also changes eligibility and decays $P_b$. Independent
probe conditions therefore need to start from copies of the same frozen state.

A zero `modulatory_drive` removes the positive structural-write term, but replay can
still update $P_b$ and structural decay (plus any configured noise) still operates.
Because an empty replay-ID list invokes the default selection, it does not encode a
replay-free pass. E022's ``null'' condition is therefore a zero-modulation condition.
Its damage intervention raises the selected branch's structural decay rate. A new
protocol must state replay selection, modulation, decay, and noise separately.

The update order can be read as six operations:

1. A cue sets fast and slow access, computes branch activation $x_b$, increments the local eligibility tag $E_b$, and passively decays $P_b$.
2. A separate consolidation call chooses replayed traces and computes branch overlap $r_b$ from their allocation weights.
3. Replay and sleep update consolidation support $P_b$; an optional global budget can then rescale the support values.
4. The slow state $M_b$ is written from the pre-decay eligibility, updated support, and nonnegative modulatory drive, with passive decay and configured noise also applied.
5. Eligibility decays after it has contributed to the structural update.
6. Recall uses the activity already gated by access; the separate linking diagnostic uses allocation weights and $M_b$ directly.

This sequence is the executable model tested in the paper. It is not a calibrated mapping
from a simulator pass to biological time, and the internal $M_b$-based diagnostic is not
the E025 common behavioral readout.

#figure(
  kind: "table",
  supplement: "Table",
  caption: [Symbols, executable updates, scale, and interpretation in the primary simulator. Code fields name the implemented variables; biological interpretations are phenomenological.],
  wide-table(
    clean-table(
      columns: (0.55fr, 1.2fr, 2.2fr, 0.9fr, 1.2fr, 1.2fr),
      inset: 3pt,
      header: ([*Symbol*], [*Mathematical role*], [*Code field/update*], [*Scale/time*], [*Biological interpretation*], [*Observability*]),
      [$x_b$], [Cue-evoked activity after access gating.], [`BranchState.activation = effective_access * cue_drive` in `apply_cue`; no temporal integration.], [Scalar per cue call; no physical time unit.], [Branch activity proxy.], [Exported simulator trace; no direct biological calibration.],
      [$A_b^f$], [Fast access from cue, context, spines, and inhibition.], [`BranchState.fast_access = sigmoid(fast_term)` in `apply_cue`.], [Bounded, dimensionless.], [Momentary local opening.], [Computed internal state; no direct measurement in this study.],
      [$A_b^s$], [Slow access derived from structural state.], [`BranchState.slow_access = sigmoid(structural_gain * M_b)`.], [Bounded, dimensionless.], [Persistent bias on branch recruitment.], [Computed internal state; `M_b` is latent.],
      [$E_b$], [Recent-activity eligibility tag.], [`branch.eligibility.value`; accumulated on cue calls, used and decayed after each consolidation update.], [Clipped to $[0,1]$; per call.], [Eligibility-like local tag.], [Exported simulator trace; not a measured biological eligibility trace.],
      [$P_b$], [Replay/sleep-dependent consolidation support.], [`branch.translation_readiness.value`; replay update, cue decay, then optional global `translation_budget` normalization.], [Clipped to $[0,1]$; per update.], [Local capture/resource availability.], [Exported resource proxy; not a protein or mRNA assay.],
      [$M_b$], [Bounded slow structural accessibility.], [`branch.structural.accessibility`; tag × support × nonnegative modulation, decay, optional Gaussian noise, then clipping.], [$[0, M_"max"]$; per consolidation pass.], [Phenomenological slow branch-local persistence.], [Exported latent state; no molecule identified.],
      [$r_b$, replay], [Replay-weighted branch overlap, optionally with neighbor spillover.], [`_replay_overlap`; selected `replay_trace_ids`, allocation weights, priorities, adjacency.], [Dimensionless overlap in $[0,1]$.], [Replay coverage of a branch.], [Known simulator schedule/input, not direct observation here.],
      [$W$, `modulatory_drive`], [Global nonnegative multiplier on structural writing.], [`max(0.0, window.modulatory_drive)` in `run_consolidation`.], [Dimensionless per pass.], [Generic permissive consolidation drive.], [Set by protocol; not identified with dopamine or another measured signal.],
      [$R_mu$], [Pre-threshold support for trace $mu$.], [`RecallSupport.support = sum(weight * branch.activation)`, followed by a context-mismatch factor when both context labels are present.], [Output of one cue/readout.], [Recall-support proxy.], [Simulator output; biological behavior would require a separate operationalization.],
      [$L_(mu nu)$], [Allocation-weighted shared structural accessibility.], [`linking_score = sum(a_mu_b * a_nu_b * M_b)`.], [Structural diagnostic; allocation scale.], [Shared slow-access route.], [Computed from latent `M_b`; not an independent observable across models.],
    ),
    extra-margin: 0.75in,
  ),
) <tab-variable-map>

Supplementary Information gives the broader theoretical motivation separately from this
code-exact discrete form. The model architecture is summarized in @fig-model-schematic.

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
    *(C)* Schematic variable roles; exact discrete updates and code mapping are in the text.
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
    [Recent activity may bias later reuse of overlapping branches beyond fixed allocation.],
    [E025 shared-branch versus source-private probe; internal traces.],
    [Full-model D is positive, but Hebbian and resource-only models produce larger D.],
    [Simulator result; the primary readout is small and not biologically calibrated.],
    [Fast-vs-slow separation],
    [A common probe tests whether slow writing adds a distinct output-level prediction.],
    [E025 equal-budget tuning and locked holdout; E017 mechanism traces.],
    [Full model exceeds five comparators, is equivalent to shuffled replay, and is exceeded by two.],
    [Eligibility-only had no qualifying tuned candidate; adapters represent specified implementations.],
    [Perturbation-sensitive linking],
    [Shared-route perturbation should harm linking more than single-trace recall.],
    [Focal overlap damage and targeted rescue.],
    [Linking drops after overlap damage and recovers most strongly after targeted rescue.],
    [Damage sensitivity alone is not unique to slow writing.],
    [Comparator discrimination],
    [Determine whether the full model has a distinct common-readout effect.],
    [E025 with frozen probes, readout, candidate budget, and holdout.],
    [No unique full-model advantage; Hebbian and resource-only models exceed it on D.],
    [A simulator comparison, not biological validation.],
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

== 1. Equal-budget tuning and holdout do not support a unique slow-writing advantage

E025 compared ten simulator families under the same one-hot cue and probe schedule,
sigmoid readout, and shared-branch versus source-private contrast. Each family received
32 prespecified parameter configurations. Candidate selection used 40 development and
40 validation topology draws; the locked comparison used 100 new holdout draws. The four
primary topology strata (canonical, strong-overlap, chain-overlap, and sparse-random) were
weighted equally within each draw. `hub_overlap` was analyzed separately as a sensitivity
case. Candidate adapters were fixed before evaluation, and no holdout outcomes were used
for tuning.

Nine families produced a qualifying tuned candidate. Eligibility-only produced none that
met the preregistered private-cue baseline and functional criteria, so its nominal result
is descriptive and no confirmatory comparison is available. For the qualifying families,
the tuned full model had immediate $D=0.02270$ (95% bootstrap CI 0.02254--0.02286).
Hebbian weight-only had $D=0.09330$ (0.09216--0.09445), and resource-only had
$D=0.09435$ (0.09309--0.09564). In paired holdout contrasts, the full-minus-Hebbian
difference was $-0.07060$ (95% CI $-0.07162$ to $-0.06959$), and the full-minus-resource
difference was $-0.07165$ (95% CI $-0.07282$ to $-0.07048$). The preregistered
sign-flip procedure classified both as comparator superiority after Holm correction.
That test assumes sign-exchangeability of paired topology-draw differences at the null;
model signs were not randomized. A post-hoc centered cluster-bootstrap sensitivity
analysis reproduced the immediate decisions for all testable families (S2 Appendix);
it holds the selected candidates fixed, is approximate and conditional on independent
draws from the registered topology generator, and was not preregistered. The full model exceeded five other testable
families and was practically equivalent to shuffled replay under both analyses. These
outcomes show that the tested slow-writing model does not
have a unique advantage on this readout. They do not show that every implementation of
Hebbian or resource-based learning would outperform it.

After nine passive-decay steps without replay or new writing, the full model's secondary
effect was $D_"late"=0.01795$ (95% CI 0.01780--0.01810). Hebbian weight-only
($0.09294$, 0.09181--0.09408) and resource-only ($0.07163$, 0.07100--0.07227) again
exceeded the full model under the corrected comparison. This persistence endpoint does
not replace the immediate primary outcome. The separate hub-overlap sensitivity result
is reported in the E025 output files and is not included in the four-stratum macro-effect.

All reported scores are dimensionless simulator readouts, not empirical probabilities
or behavioral effect sizes. No biological dataset was used to tune or evaluate the E025
primary comparison. The preregistration, candidate registry, topology and probe
manifests, run code, and full holdout outputs are routed in S2 Appendix and the Data and
Code Availability section.

#figure(
  image("figures2/Fig_e025_01_tuned_common_probe.png", width: 100%),
  caption: [
    *E025 tuned common-probe comparison.*
    (A) Immediate $D$ for each qualifying tuned model on 100 locked holdout topology
    draws; points are means and bars are 95% bootstrap intervals. Eligibility-only is
    shown as unavailable because no candidate met the preregistered baseline and
    functional criteria. (B) Paired $D_"full"-D_"comparator"$ contrasts
    with 95% bootstrap intervals. Blue marks full-model superiority, gray marks practical
    equivalence, and red marks comparator superiority after Holm correction. The shaded
    region is the prespecified ±0.02 practical equivalence margin. The full model exceeded
    five comparators, was practically equivalent to shuffled replay, and was exceeded by
    Hebbian weight-only and resource-only models. Readouts are simulator scores, not
    empirical probabilities.
  ],
) <fig-e025-tuned-probe>

== 2. What the simulator produces

=== Traceable internal dynamics make the slow-writing hypothesis executable

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

=== Canonical mechanism signatures describe the full model's internal profile

The canonical model was evaluated using five protected signatures:

- *SIG-A:* overlap-branch structural writing,
- *SIG-B:* linking gain after consolidation,
- *SIG-C:* context separation,
- *SIG-D:* linking-vs-recall dissociation under overlap damage,
- *SIG-E:* targeted rescue selectivity.

The full model passed all five signatures under the locked canonical protocol.
Overlap-branch structural writing was positive, the $M_b$-based linking diagnostic
increased after consolidation, context separation was present, linking was more
vulnerable than single-trace recall under overlap damage, and targeted rescue produced
strong normalized recovery relative to generic no-precue consolidation.

These signatures describe the full model's internal behavior under its canonical
protocol. Several depend directly on $M_b$, so they do not establish an output-level
advantage over models without that state. SIG-C is largely architectural and
context/allocation-related, while SIG-D can arise from overlap-branch perturbation
geometry.

The perturbation-and-rescue logic behind these canonical signatures is illustrated in
Supplementary Figure S5.6 (S5 Appendix).

#figure(
  kind: "table",
  supplement: "Table",
  caption: [Canonical mechanism-signature profile under the locked reference protocol. These simulator diagnostics describe the full model and do not rank model families on E025's common readout.],
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

Taken together, the canonical run shows overlap strengthening, an increase in the
$M_b$-based linking diagnostic, perturbation-sensitive loss, and targeted recovery
(@tab-canonical-signatures). These are simulator-internal results, not evidence that the
mechanism is required or biologically validated.

=== Baseline ablations characterize the signature definitions

We next compared the full model against baseline comparators:

- *fast-context-only,*
- *replay-without-structure,*
- *random slow drift,*
- *fixed allocation only.*

Context separation passed in several comparator models, showing that this result can
arise from fast gating or allocation structure without the slow update. SIG-A, SIG-B,
and SIG-E, however, include $M_b$-dependent quantities. A comparator with structural
writing disabled cannot pass those quantities by construction. E018 therefore checks
the behavior of the implemented signatures under mechanism ablations; it does not show
that those models fail a shared recall-based outcome.

The E018 results also show that fixed overlap geometry, unstructured drift, and
fast-context gating affect different parts of the signature profile. These are useful
mechanistic diagnostics, but do not decide whether an alternative can match the full
model under one common probe and readout.

=== Historical hard-comparator results use mechanism-specific proxies

To make the comparator test fairer, we added harder alternatives:

- *Hebbian weight-only,*
- *soma/global-gain-only,*
- *shuffled replay,*
- *eligibility-only,*
- *resource-only.*

E022 reports $M_b$-dependent structural diagnostics alongside B1--B7 values. The B1
proxy varies by comparator: it uses structural linking for some models and accumulated
weight, eligibility, or resource values for others. B3 uses the $M_b$-based structural
linking score even for models without an active $M_b$ write. For Hebbian, eligibility,
and resource comparators, the damage and rescue branches reuse the pre-damage behavioral
proxy, so B5 is zero by construction. That value does not measure those models' ability
to recover under a shared recall probe.

The E022 files remain useful for inspecting how the chosen mechanisms behave in the
implemented simulator. Their B1/B3/B5 matrix does not apply one readout to every model,
and its classifications cannot establish that no alternative matches the full model on
common recall or intervention outcomes. In three of four tested motifs, gSIG-B is
undefined because no unlinked pairs exist; the stored `NaN` values are not failures.

#figure(
  image("figures2/Fig_e023_03_comparator_matrix.png", width: 100%),
  caption: [
    *Historical E018/E022 comparator summary.*
    Left: mechanism-signature labels; right: the B1/B3/B5 proxies assembled for E022.
    The source metrics use different definitions across model families. Colors show the
    legacy thresholds and are not a common model-selection result. `N/A` denotes a
    measure without a defined denominator. E025 provides the separate common-probe
    comparison reported in Fig. 2.
  ],
) <fig-comparator-matrix>

#figure(
  kind: "table",
  supplement: "Table",
  caption: [Historical E018/E022 summary. Structural and output columns draw on different protocols and, for some comparators, different score definitions. The last column records whether these files establish a common behavioral ranking.],
  clean-table(
    columns: (1.3fr, 1.7fr, 1.3fr, 1.7fr, 0.8fr),
    header: ([*Comparator*], [*Mechanism removed or replaced*], [*Structural summary*], [*Reported output proxy*], [*Common-outcome inference*]),
    [full_model], [All mechanisms active.], [SIG-A = 0.128; SIG-B = 0.240; gSIG-E = 1.57.], [B1 = 0.239; B3 = 0.141; B5 = 1.57.], [Not established],
    [fast_context_only], [Fast gating retained; $M_b$ frozen.], [SIG-A/B/E = 0; only SIG-C and SIG-D pass.], [Context separation and damage sensitivity remain architecture-driven.], [Not established],
    [replay_no_structure], [Replay without persistent write.], [SIG-B = -0.018; SIG-E = 0.], [Transient replay is insufficient for durable linking.], [Not established],
    [random_slow_drift], [Non-specific slow drift with matched scale.], [SIG-A = -0.089; SIG-B = -0.065.], [Produces non-specific or unstable effects.], [Not established],
    [fixed_allocation_only], [Static overlap without dynamic update.], [SIG-A/B/E = 0.], [Preserves context geometry but no replay-written gain.], [Not established],
    [hebbian_weight_only], [Weight updates only.], [gSIG-A = 0 across tested motifs.], [Canonical B1 = 4.18, but B5 = 0.], [Not established],
    [soma_global_gain_only], [Branch-independent gain.], [gSIG-A = 0; chain gSIG-B = 0.140.], [Canonical B1 = 0.198; B5 = 1.88 without specificity.], [Not established],
    [eligibility_only], [Transient eligibility without retained structure.], [gSIG-A = 0.], [Canonical B1 = 0.398; B5 = 0.], [Not established],
    [resource_only], [Capture resources without retained $M_b$.], [gSIG-A = 0.], [Canonical B1 = 0.406; B5 = 0.], [Not established],
    [shuffled_replay], [Replay identity scrambled.], [gSIG-A = 0.091 at $n$=4 but decays with scale.], [Canonical B1 = 0.191; B5 = 1.82, but specificity collapses.], [Not established],
  ),
) <tab-comparator-summary>

E018/E022 remain descriptions of their original protocols. E025 supplies the tuned,
held-out common-probe result; the historical signature and proxy classifications are not
additional evidence that slow structural state is necessary for these effects.

== 3. Where the model generalizes and where it fails

=== One-at-a-time robustness identifies bounded functional ranges

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

=== Two-parameter heatmaps reveal a write--replay regime and timing boundary

We then tested biologically meaningful parameter interactions. Six parameter pairs were
evaluated:

- structural learning rate × replay gain,
- eligibility decay × timing gap,
- overlap strength × replay gain,
- structural decay × structural learning rate,
- structural noise × replay gain,
- context gain × structural learning rate.

The structural learning rate × replay gain heatmap showed a coherent write--replay
regime for the full model's internal signature profile. In this tested grid, low
structural learning rate or low replay gain caused that profile to fail. The eligibility
decay × timing gap heatmap showed a temporal boundary:
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
    [structural_lr × replay_gain], [70 of 99 cells passed (70.7%).], [Low write rate or low replay collapses SIG-A, SIG-B, and SIG-E.], [Both shape the full model's internal profile in this tested grid.],
    [eligibility_decay × timing_gap], [45 of 72 cells passed (62.5%).], [Long delays with rapid eligibility decay fail.], [Tag lifetime limits consolidation timing.],
    [overlap_strength × replay_gain], [37 of 72 cells passed (51.4%).], [Overlap below about 0.40 fails even with replay.], [The full model's internal profile requires sufficient overlap in this tested grid.],
    [structural_noise × replay_gain], [55 of 63 cells passed (87.3%).], [Extreme noise with weak replay destabilizes the profile.], [Moderate noise tolerance is broad but not unlimited.],
  ),
) <tab-robustness-summary>

The robustness analyses therefore support a bounded functional regime rather than a
single tuned point (@fig-robustness-landscape, panel B–C; @tab-robustness-summary).
The interpretive limit is that these heatmaps cover selected parameter pairs, not
arbitrary high-dimensional combinations.

=== Scaling and motif tests show topology-dependent limits

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

=== Shuffled replay loses specificity as branch allocation space expands

The hard-comparator analysis showed a small apparent structural signal for shuffled
replay in the smallest four-branch motifs. We therefore audited shuffled replay across
branch counts and seeds.

With 20 seeds, the apparent effect was shown to be a high-end small-sample draw rather
than a stable signal. In the four-branch motif, shuffled replay reached only about 7% of
the full-model overlap advantage on average. At 8 branches the ratio fell below 5%, and
by 16--32 branches it approached zero. Variance also decreased with increasing branch
count.

Under the E022R structural-specificity metric, the tested shuffled-replay variant did not
preserve the full model's scaling profile. This result is specific to that variant and
internal measure; E025 did not establish a general requirement for identity-preserving
replay across broader model families.

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
    Take-home: this shuffled-replay variant loses structural specificity as the tested
    branch allocation space expands.
  ],
) <fig-shuffled-replay>

Shuffled replay can weakly mimic overlap writing in the smallest allocation space by
chance, but this effect decays rapidly with branch count (@fig-shuffled-replay). The
result concerns the implemented replay-identity disruption tested here, not every
possible replay-disruption model.

// ─── Discussion ──────────────────────────────────────────────────────────────
= Discussion

== What the simulator supports

The simulator provides a reproducible, branch-resolved implementation of a slow-access
hypothesis, with inspectable internal states and bounded parameter and topology behavior.
E025 gives each model family the same tuning budget and evaluates selected candidates on
new topology draws. On the common shared-branch probe, the tuned full model exceeded five
alternatives, was practically equivalent to shuffled replay, and was exceeded by the
Hebbian weight-only and resource-only models. Eligibility-only did not produce a
qualifying candidate and therefore has no confirmatory comparison. The result is not a
unique advantage for slow writing; it is a bounded model-comparison finding about the
specified implementations, adapters, task, and readout.

The contribution is an executable hypothesis and an explicit account of where its
output-level predictions do and do not distinguish it from alternatives. The earlier
SIG-A--SIG-E profile remains a description of the full model's internal behavior, not
evidence of unique cross-model performance. Context separation and perturbation-sensitive
linking are not diagnostic of slow writing on their own. Robustness and motif analyses
define additional simulator boundary conditions; they do not reverse the E025 comparison.

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
structures. Published work already supports a role for branch-local dendritic plasticity
in memory linking; our study does not claim that general mechanism as novel. It uses
hippocampal memory linking as a model testbed for the narrower question of whether this
particular slow latent state adds explanatory value over tuned alternatives
@Choucry2024MemoryLinkingIdentity @Kastellakis2023DendriticEngram @Sehgal2025ContextLinking
@Guskjolen2023EngramNeurons @Zaki2024OfflineLinking, but similar fast/slow access
separations may be relevant in prefrontal contextual control
@WangYang2018Routing @Keller2020ContextualModulation @Bos2025GainModulation
@Olah2025HCNGating, working memory stabilization, and systems-level consolidation
@Frankland2005RecentRemoteMemories @Wang2024SleepDependentEngramReactivation
@TononiCirelli2020SleepPlasticity @ReyesResina2021SleepConsolidation. Those extensions
remain speculative until separately implemented and tested.

The shuffled-replay audit shows that, under its structural-specificity measure, the
implemented model's scalable specificity depends on replay preserving branch identity.
This is a simulator-level constraint on that implementation. E025 did not show a
distinct full-model advantage on its common recall readout, so this result does not
establish that replay identity is required across broader model classes.

== Failure modes are informative

Several failures strengthen the model's interpretability. Weak overlap fails. Excessive
hub overlap produces over-linking. Long timing gaps fail when eligibility decays. High
structural decay erodes persistence. Shuffled replay fails as allocation space expands.
These failures show that the model is not a free-floating explanation that succeeds
everywhere. It defines boundary conditions.

== Future tests

The next computational test should compare multiple independently justified adapters
within the Hebbian, eligibility, resource, and slow-structure families, then repeat the
shared readout and intervention comparison on a new locked holdout. E025 uses explicit,
reproducible adapters but represents only one implementation per broad mechanism family;
its result cannot be generalized to every model in those families.

The most direct future biological tests would require longitudinal branch-resolved
recordings across linked memory formation, replay, perturbation, and recovery. They
should dissociate fast contextual gating from slower replay-dependent stabilization and
test whether linking is more vulnerable than single-trace recall under a measured,
targeted dendritic intervention @Sehgal2025ContextLinking @Kastellakis2023DendriticEngram
@Uytiepo2025EngramArchitecture.

Future simulator work should extend beyond branch-count scaling toward more biologically
structured dendritic trees, including proximal/distal organization, apical and basal
compartments, spine distributions, inhibitory gating, branch-specific thresholds, and
hippocampus--prefrontal differences.

// ─── Materials and Methods ───────────────────────────────────────────────────
= Materials and Methods

== Overview

The study is organized as a simulator-first model comparison. E025 is the primary
equal-budget tuned comparison of ten simulator families under a shared probe and
readout, with candidate selection on development and validation draws and a locked
holdout. E017--E022R provide simulator traces, internal diagnostics, comparator history,
and parameter/motif analyses; E024 is the earlier common-probe analysis that informed
the E025 protocol. E025 found that the full model did not uniquely outperform the tested
alternatives. Exploratory open-data bridges are not part of the primary evidence ladder
and are described separately in S3 Appendix.

== Study design and evidence tiers

The study contains three explicit evidence tiers. The primary evidence tier contains the
simulator experiments `E017--E025` together with the assembled article figures,
as listed in @tab-experiment-registry. E017--E022R provide the historical mechanism,
robustness, and motif evidence; E024 established the common probe, and E025 applies
equal-budget tuning and the locked cross-model holdout protocol.
A second legacy tier contains earlier experiment
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
    [E025], [Equal-budget tuned common-probe comparison], [Fig. 2],
    [E024], [Historical common-probe comparison and protocol development], [S2 Appendix],
    [E017], [Trace export and canonical mechanism signatures], [Fig. 3; Table 2],
    [E018], [Baseline comparator matrix], [Fig. 4; Table 3],
    [E019], [One-at-a-time robustness], [Fig. 5; Table 4],
    [E020], [Two-parameter heatmaps], [Fig. 5; Table 4],
    [E021], [Scaling and motif generalization], [Fig. 6; Table 5],
    [E021R], [Specificity gate and motif language], [Fig. 6 interpretation],
    [E022], [Historical hard comparators], [Fig. 4; Table 3],
    [E022R], [Shuffled replay audit], [Fig. 7],
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
and `E022R` add historical hard comparators and the shuffled-replay specificity audit.
`E024` introduced the common-probe comparison. `E025` runs the preregistered tuning and
locked-holdout analysis. Full
computational routing, software/resource identifiers, and reproducibility details are
collected in S4 Appendix.

== Simulator architecture

The canonical simulator contains four branches and two traces. One branch is shared
between the traces and serves as the overlap branch; each trace also has private
branches. `apply_cue` computes fast access, slow access, and access-gated activation in
one discrete call. It accumulates eligibility and passively decays consolidation support.
It does not integrate branch activity over time or deliver replay as an activity input.
`run_consolidation` separately selects replayed traces, computes priority-weighted branch
overlap (plus optional neighbor spillover), updates `P_b`, applies an optional global
resource budget, updates and clips `M_b`, and then decays `E_b`. A call is one update
under a dimensionless simulator scale; the code does not assign it minutes or hours.

The structural linking score $L_(mu nu)$ is computed from fixed trace allocations and
`M_b`. Recall support instead sums each trace's allocation weights times the activity
already gated by effective access. The two outputs answer different questions, and the
structural score is not a common behavioral readout for a comparator that has no `M_b`.
As shown in @tab-variable-map, a probe implemented by `apply_cue` also changes `E_b` and
`P_b`; independent probe conditions must therefore use copies of a shared frozen state.

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
condition. E024 uses paired technical seeds `100--149`; E025 uses 40 development, 40
validation, and 100 holdout topology draws. Bootstrap confidence intervals and paired sign-flip
tests resample or operate over the 100 independent holdout draws. These are technical simulator units rather than
biological replicates.

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
E022 evaluated structural signatures and output proxies with model-specific definitions,
so it remains diagnostic history rather than a common model-selection test. E024
established a shared probe and sigmoid readout; E025 adds equal-budget tuning, separate
development and validation partitions, and a locked topology holdout. The results apply
to the frozen adapters and parameter search space, not to every implementation of each
broad mechanism family.

== Primary common-probe model comparison (E025)

For each eligible directed trace pair $mu -> nu$, let
$d_t = p_nu^t("shared") - p_nu^t("private")$. The primary outcome is
$D_(mu -> nu) = d_"post" - d_"pre"$. Each $p_nu$ is the common sigmoid readout for the
target trace after a one-hot cue to the shared or source-private branch.

Each probe starts from a separate copy of the same frozen pre- or post-consolidation
state. The shared output equation is `s_j = sum_b W_jb h_b x_b`, followed by the same
fixed sigmoid readout. Model-specific weights and branch-gain adapters are defined by
the frozen E025 registry. The readout is a dimensionless simulator score on a 0--1
scale, not an empirical probability. The same probe manifest and sigmoid settings are
used for all families.

Each of the ten families received 32 unique prespecified parameter configurations.
Candidates were ranked on 40 development topology draws; up to eight per family were
checked on 40 independent validation draws. The selected candidate was then evaluated
once on 100 previously unopened holdout draws. The four primary strata are canonical,
strong-overlap, chain-overlap, and sparse-random; each receives equal weight within a
draw. The separate hub-overlap sensitivity condition is not included in this macro
effect. All families share the same generated topologies and paired dynamics streams.

Before candidate selection, models had to match the full-model nominal private-cue
baseline within ±0.02, retain private-cue recall after consolidation, and preserve correct
over incorrect private-cue discrimination. Candidate adapters and these criteria were
not changed after holdout outcomes were inspected. Nine families yielded a qualifying
tuned candidate. Eligibility-only did not pass the development and validation filters;
its nominal holdout result is descriptive only and no confirmatory comparison is made.

Confidence intervals use 10,000 bootstrap resamples of holdout topology draws. The 95%
percentile intervals are pointwise and are not adjusted across the 54 hypotheses; Holm
correction applies to hypothesis tests. Paired model contrasts use 100,000 sign-flip
tests. The prespecified practical margin
is ±0.02 on the paired difference `D_full - D_comparator`, not on the absolute model
means. Full-model superiority, comparator superiority, and practical equivalence are
tested as separate outcomes with a joint Holm correction across 54 hypotheses (nine
comparators, two time points, and three outcomes). Comparisons that do not support one of
these claims remain inconclusive; a missing qualifying candidate is reported as
unavailable, not as a win, loss, or equivalence. The registered sign-flip tests
assume sign-exchangeability of paired draw-level differences at each null boundary; signs
were not randomized by the simulator design. S2 Appendix reports an explicitly post-hoc,
centered topology-cluster bootstrap sensitivity analysis with the same hypothesis family.

The secondary persistence endpoint repeats the probe after nine passive-decay steps
without cue input, replay, or new writing. Eligibility and readiness decay passively and
the structural state undergoes its configured passive decay. This endpoint cannot
replace the immediate primary comparison. Holdout draw-level macro scores, candidate-
selection results, paired contrasts, and the hub sensitivity output are saved
under `results/e025_common_probe_tuning/`; the preregistered plan and full topology and
probe manifests are listed in S2 Appendix.

== Historical common-probe analysis (E024)

E024 used 50 paired technical seed clusters and established the shared probe and readout
later retained for E025. Its original results are preserved as historical protocol
development and are not combined with the independent E025 holdout estimates. E024
outputs remain available under `results/e024_common_probe_comparison/`.

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

The E017--E023 analyses include deterministic reference runs and predeclared mechanism
signatures. E024 adds a frozen common-probe protocol and paired technical seed clusters;
it does not fit model parameters to the primary outcome. Its calibration is restricted
to the pre-consolidation private-cue readout, and published-parameter results are also
reported.

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
used to tune or validate the E025 primary comparison. They do not directly measure $M_b$,
dendritic branch accessibility, cytoskeletal writing, or a molecular memory code.
Table @tab-open-data-summary identifies the analyzed bundles and units; S3 Appendix gives
the dataset-specific pipelines and null constructions.

#figure(
  kind: "table",
  supplement: "Table",
  caption: [Public datasets used only for supplementary exploratory observable bridges. None measures the latent state $M_b$ or contributes observations to E025 tuning or holdout evaluation.],
  clean-table(
    columns: (0.9fr, 1.8fr, 1.8fr, 2.1fr),
    header: ([*Archive*], [*Analyzed bundle*], [*Measurement*], [*Question and boundary*]),
    [`DANDI 000718`], [3 NeutralExposure-to-offline session pairs], [Registered ROI ensemble activity and offline events], [Offline core-unit enrichment; exploratory, not a direct branch or $M_b$ measure @Zaki2024DANDI000718],
    [`DANDI 000336`], [6 NWB files in 3 within-session plane pairs], [Paired imaging-plane activity; 2 cross-depth pairs and 1 cross-area pair], [Cross-plane coupling relative to within-plane and time-shift nulls @Amaya2026DANDI000336],
    [`DANDI 001710`], [23 subjects (7 Cre, 9 Ctrl, 7 SparseKO); 139 NWB files], [Longitudinal hippocampal population activity with behavior], [Place-code continuity and perturbation-group comparison; exploratory and indirect @Plitt2026DANDI001710],
  ),
) <tab-open-data-summary>

// ─── Data and Code Availability ──────────────────────────────────────────────
= Data and Code Availability

Code availability: the public `cytodendaccessmodel` GitHub repository is at
`https://github.com/NoeticDiffusion/cytodendaccessmodel`. The existing Zenodo snapshot
(DOI `10.5281/zenodo.20813821`) predates E025 and should be treated as an earlier code
release. The E025 runner, figure renderer, preregistration, parameter registry, topology
and probe manifests, and holdout outputs are included with this manuscript's source
files under `experiments/`, `project/neural_computation/`, and
`results/e025_common_probe_tuning/`. A versioned public archive containing this complete
E025 package must be deposited and cited before journal submission. The code is released
under the GNU General Public License v3.0 (`GPL-3.0`).

Data availability: primary simulator outputs supporting the main evidence ladder are
organized under `results/e025_common_probe_tuning`, `results/e024_common_probe_comparison`,
`results/e017_traceable_simulator_core`,
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
and manuscript preparation. These tools are not authors and did not provide peer review. The author
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
  scaling analyses, specificity metrics, shuffled replay audit, E025's tuned common-probe
  adapters and outcome, and E024's historical common-probe protocol. This appendix
  contains reproducibility details for E017--E025 at the simulator-design level.
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

#supplement-box([S6 Changelog. Internal record of manuscript changes])[
  Internal record of changes to the manuscript and supporting materials, included for
  document history only. It is not part of the article's evidence or claims.
]
