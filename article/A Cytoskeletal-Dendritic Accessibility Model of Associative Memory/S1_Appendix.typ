#import "template.typ": essay-template

#show: doc => essay-template(
  short_title: [S1 Appendix],
  doc,
)

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

This makes explicit that synaptic couplings provide a baseline landscape, while slow structural accessibility deforms which basins are easiest to enter or stabilize. We treat this as optional because it abstracts away from much of the biology emphasized in the main text.

== Executable Bridge And Claim Classes

The executable section of the present unified manuscript helps sharpen the epistemic status of several claims made in the biological theory. In particular, it distinguishes three classes of consequence that are easy to blur in prose alone.

First, some consequences are largely architectural. In the executable model, context-sensitive disambiguation can already arise from the fast contextual layer, and linking is more fragile than single-trace recall under focal overlap damage partly because the linking metric weights shared branches multiplicatively. These are still meaningful results, but they should be read as consequences of the formal architecture rather than as uniquely diagnostic evidence for slow structural rewriting.

Second, some consequences become mechanistically diagnostic only after executable perturbation. Comparator baselines showed that overlap-branch strengthening, linking growth after consolidation, and selective rescue of association disappear when replay-specific slow structural writing is removed. These signatures therefore track the slow structural layer more specifically than context sensitivity or overlap geometry alone.

Third, the broader overlap-motif analysis showed that the theory is not confined to a single hand-built two-trace example. Linking scaled with structured overlap across weak, chain, strong, and hub motifs, while weak overlap marked a boundary condition below which a shared branch does not automatically become a structural hub. This is useful for the present theory because it suggests that the prediction is gradated: shared structural allocation should matter in proportion to how strongly replay and allocation repeatedly recruit the same dendritic subunits.

The executable bridge also clarifies one formal point. In the biological formalization, recall support is written schematically as $R_mu(t) = sum_b a_(mu b) A_b(t) x_b(t)$ to make the role of accessibility explicit. In the simulator, the same accessibility factor is absorbed earlier into branch activity, so recall support is computed from access-gated $x_b$ rather than by multiplying accessibility twice at readout. The conceptual role is the same: slow structural accessibility still biases which branches can participate in recall. The difference is implementation order, not theoretical meaning.

#table(
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
)

#par(first-line-indent: 0pt)[
  #emph[Table 4. Theory-to-executable mapping for recall-support and access variables.]
]

#bibliography("references_cytoskeletal_dendritic_accesibility_model.bib")
