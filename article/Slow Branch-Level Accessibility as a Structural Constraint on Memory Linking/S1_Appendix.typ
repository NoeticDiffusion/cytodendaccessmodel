#import "template.typ": essay-template

#show: doc => essay-template(
  short_title: [S1 Appendix],
  doc,
)

// Tables in this appendix are numbered S1.1, S1.2, …
#set figure(numbering: (..num) => "S1." + str(num.at(-1)))

= S1 Appendix. Mathematical framework and executable bridge

== Minimal State-Space Formulation

For readers who prefer a compact technical summary, the present hypothesis can be expressed as a two-timescale state-space model:

$ A_b(t) = A_b^f(x_b(t), s(t), C(t)) A_b^s(M_b(t)) $

$ dot(x)_b = A_b(t) F_b(x, I, s) $

$ dot(s)_i = G_i(s, x, M) $

$ dot(M)_b = eta E_b(t) sigma(delta(t) - theta_delta) (1 - M_b(t) / M_"max") - lambda_M M_b(t) + sqrt(2 T_("eff")) xi_b(t) $

$ R_mu(t) = sum_b a_(mu b) A_b(t) x_b(t) $

The interpretation is:

- $x_b$: fast dendritic integration state,
- $s_i$: local spine/synapse accessibility,
- $A_b$: effective branch accessibility,
- $M_b$: slow branch-level structural state,
- $E_b$: eligibility trace,
- $a_(mu b)$: branch allocation for trace $mu$,
- $R_mu$: support for retrieval of memory trace $mu$.

In this formulation, the hypothesis does not require that cytoskeletal variables store memory content directly. Instead, they modulate the accessibility structure over which memory traces are encoded and later retrieved. The quantity $R_mu$ should be read as pre-threshold recall support; a later nonlinear readout or recurrent completion stage may convert support into full pattern completion.

== Phenomenological interpretation of the write terms

Two terms in the slow update deserve explicit biological framing because they are phenomenological rather than molecule-specific. The delayed threshold $theta_delta$ should be read as a coarse write-enable condition: only when modulatory, biochemical, or replay-linked consolidation signals exceed a local threshold does recent eligibility get converted into a durable branch update. This could correspond, in compressed form, to saturation-like requirements on calcium/CaMKII signaling, synaptic-capture conditions, or other delayed plasticity-permissive gates rather than to a single directly measured concentration @Yagishita2014DopamineTiming @RedondoMorris2011STC @Gerstner2018EligibilityTraces @Rogerson2014SynapticTaggingAllocation @Ibrahim2024STC.

The capacity term $M_"max"$ plays a different role. It is not meant to imply a hard count of microtubules, spines, or proteins. Instead, it provides a bounded phenomenological surrogate for the fact that branch-local structural support is finite: transport resources, physical occupancy, local translation capacity, and active-matter-like nonequilibrium constraints all limit how much accessibility can be durably stabilized in one compartment before turnover and competition dominate @fodor2016 @needleman2017 @Govindarajan2011DendriticBranch @Rangaraju2019MitoCompartments @Das2023LocalTranslationMemory @Hacisuleyman2024DendriticTranslation. In other words, $M_"max"$ is a reviewer-facing way to keep the model from smuggling in unlimited structural write capacity.

== Factorized Accessibility View

The central formal move in the paper is the factorization

$ A_b(t) = A_b^f(t) A_b^s(t) $

where $A_b^f$ summarizes faster dendritic, contextual, or circuit-level opening, and $A_b^s$ summarizes slower structural accessibility. This is the cleanest mathematical way to state the paper's USP: memory participation depends on a slow structural gating bias beneath faster dendritic and circuit gating.

== Resource-Capture Extension

If one wishes to model local translation or branch-specific stabilization resources explicitly, introduce

$ tau_P dot(P)_b = -P_b + rho_nu nu(t) + rho_"sleep" Omega(t) r_b(t) $

where $P_b$ denotes a branch-local capture or consolidation resource, $nu(t)$ denotes a global neuromodulatory or salience-related drive, $Omega(t)$ denotes consolidation-window intensity (for example spindle- or replay-rich sleep opportunity), and $r_b(t)$ denotes branch-local replay or reactivation recruitment.

and replace the slow update with

$ dot(M)_b = eta E_b(t) P_b(t) sigma(delta(t) - theta_delta) (1 - M_b(t) / M_"max") - lambda_M M_b(t) + sqrt(2 T_("eff")) xi_b(t) $

Here $P_b$ is not itself the access state. Rather, it is a local capture or consolidation resource that helps determine whether tagged branches become durably rewritten.

== Correspondence between general and executable forms

The main text uses the executable simulator form, where replay-linked consolidation
support is represented explicitly as $P_b(t) W(t)$:

$ dot(M)_b = eta E_b(t) P_b(t) W(t) (M_"max" - M_b) - lambda_M M_b + epsilon_b(t) $

This collapses the more general $sigma(delta(t) - theta_delta)$ write-permission gate
into a binary consolidation window $W(t)$, and makes the resource-capture variable $P_b$
an explicit, traceable column in the simulator export. The general state-space form
(Eq. S1.1) and the resource-capture extension (Eq. S1.8) both reduce to this executable
form when $sigma(delta - theta_delta) approx W(t)$ and $Omega(t) r_b(t)$ approximates
the replay drive used in the main-text experiments. Reviewers comparing the main text
equations with S1 should treat Eq. S1.8 as the bridge between the biological
state-space notation and the implemented simulator.

== Branch Allocation And Memory Linking

An explicit branch-allocation view treats a trace as depending on a subset of branches:

$ a_(mu b) in [0, 1] $

with recall

$ R_mu(t) = sum_b a_(mu b) A_b(t) x_b(t) $

and a simple linking metric

$ L_(mu nu) = sum_b a_(mu b) a_(nu b) M_b(t) $

Under this view, memories formed close in time should become more linked when they reuse overlapping branches that remain in high structural accessibility states.

== Accessibility Matrix View

It is sometimes useful to summarize slow structural constraints by an effective accessibility matrix $A(M)$:

$ A_(i j)(M) $

where diagonal terms capture branch-local accessibility and off-diagonal terms summarize effective couplings induced by shared dendritic resources, correlated allocation history, or common transport constraints. This should be interpreted as an effective summary object, not as a claim that distinct branches are connected by literal microtubule wires.

== Optional Attractor-Energy View

For readers who prefer an associative-memory formulation, define

$ W_"eff"(t) = D_A(t) W D_A(t) $

where $D_A(t)$ is the diagonal matrix whose entries are the branch accessibility values.

and the energy-like quantity

$ E_"attr"(x, t) = -1/2 x^T W_"eff"(t) x + sum_b U_b(x_b) $

This makes explicit that synaptic couplings provide a baseline landscape, while slow structural accessibility deforms which basins are easiest to enter or stabilize. We treat this as optional because it abstracts away from much of the biology emphasized in the main text @Podlaski2025HighCapacity @Koch1999Biophysics @Knierim2016Tracking.

== Executable Bridge And Claim Classes

The executable section of the present unified manuscript helps sharpen the epistemic status of several claims made in the biological theory. In particular, it distinguishes three classes of consequence that are easy to blur in prose alone.

First, some consequences are largely architectural. In the executable model, context-sensitive disambiguation can already arise from the fast contextual layer, and linking is more fragile than single-trace recall under focal overlap damage partly because the linking metric weights shared branches multiplicatively. These are still meaningful results, but they should be read as consequences of the formal architecture rather than as uniquely diagnostic evidence for slow structural rewriting.

Second, some consequences become mechanistically diagnostic only after executable perturbation. Comparator baselines showed that overlap-branch strengthening, linking growth after consolidation, and selective rescue of association disappear when replay-specific slow structural writing is removed. These signatures therefore track the slow structural layer more specifically than context sensitivity or overlap geometry alone.

Third, the broader overlap-motif analysis showed that the theory is not confined to a single hand-built two-trace example. Linking scaled with structured overlap across weak, chain, strong, and hub motifs, while weak overlap marked a boundary condition below which a shared branch does not automatically become a structural hub. This is useful for the present theory because it suggests that the prediction is gradated: shared structural allocation should matter in proportion to how strongly replay and allocation repeatedly recruit the same dendritic subunits.

The executable bridge also clarifies one formal point. In the biological formalization, recall support is written schematically as $R_mu(t) = sum_b a_(mu b) A_b(t) x_b(t)$ to make the role of accessibility explicit. In the simulator, the same accessibility factor is absorbed earlier into branch activity, so recall support is computed from access-gated $x_b$ rather than by multiplying accessibility twice at readout. The conceptual role is the same: slow structural accessibility still biases which branches can participate in recall. The difference is implementation order, not theoretical meaning.

#figure(
  kind: table,
  supplement: [Table],
  caption: [Theory-to-executable mapping for recall-support and access variables.],
  table(
    columns: (auto, auto, auto),
    inset: 6pt,
    stroke: 0.5pt + black,
    align: (left, left, left),
    table.header(
      [*Theory variable*], [*Executable surrogate*], [*Same role or simplification*],
    ),
    [$A_b^f$, $A_b^s$, $A_b$], [`fast_access`, `slow_access`, `effective_access`], [same factorized access roles],
    [$M_b$], [`branch.structural.accessibility`], [same slow structural accessibility variable],
    [$E_b$], [`branch.eligibility.value`], [same eligibility / local-tag role],
    [$P_b$ or local capture state], [`branch.translation_readiness.value`], [same consolidation-support role in compressed form],
    [$R_mu = sum_b a_(mu b) A_b x_b$], [`R_mu = sum_b a_(mu b) x_b` with access-gated `x_b`], [same conceptual role; accessibility applied earlier rather than twice at readout],
  ),
) <tab-s1-theory-mapping>

#bibliography("references_slow_branch_level_accessibility.bib")
