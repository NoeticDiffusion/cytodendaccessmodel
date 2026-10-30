#import "template.typ": essay-template

#show: doc => essay-template(
  short_title: [S1 Appendix],
  doc,
)

// Tables in this appendix are numbered S1.1, S1.2, …
#set figure(numbering: (..num) => "S1." + str(num.at(-1)))

= S1 Appendix. Mathematical framework and executable bridge

== Executable discrete state updates

The E017--E025 simulator experiments use discrete updates, not the continuous
state-space equations used in some broader model descriptions. E024 introduced registered
alternative state adapters; E025 compares their output under a shared probe and sigmoid
readout after equal-budget tuning and on a locked holdout. All values are dimensionless simulator quantities.
A cue call and a consolidation pass are separate updates, with no mapping from a call to
biological seconds, minutes, or hours. The main text includes the full
symbol/code/interpretation table.

At each `apply_cue` call, let $I_b$ be cue drive, $c_b$ contextual bias, $h_b$
inhibitory tone, and $q_b$ mean spine support (1 if the branch has no spines). The
simulator computes:

$ A_b^f = sigma(g_f I_b + g_c c_b + g_"sp" q_b - g_i h_b) $
$ A_b^s = sigma(g_M M_b) $
$ A_b = op("clip")_(0,1)(A_b^f A_b^s) $
$ x_b = A_b I_b $
$ E_b^"cue" = op("clip")_(0,1)((1-d_E) E_b + abs(x_b)) $
$ P_b^"cue" = op("clip")_(0,1)((1-d_P) P_b). $

There is no recurrent input or replay-driven activity integration in `apply_cue`. Replay
selects traces in a later consolidation call. With selected set $S$, direct replay
overlap is the replay-priority-weighted mean of each selected trace's branch allocation;
it is zero if no traces are selected or their total priority is zero. Optional neighbor
spillover is added and clipped to $[0,1]$. A nonempty replay-ID list selects those IDs;
an empty list invokes the default context-compatible selection.

During consolidation, $P_b$ first accumulates replay- and sleep-linked support and is
clipped to $[0,1]$. If the total exceeds a positive `translation_budget`, every branch's
$P_b$ is scaled by the same factor. The structural update then uses this normalized
support and the current eligibility, followed by eligibility decay:

$ P_b^+ = op("clip")_(0,1)((1-d_P)P_b + g_R r_b + g_"sleep" s r_b) $
$ M_b^+ = op("clip")_(0,M_"max")(M_b + eta E_b P_b^+ max(0,W)(1-M_b/M_"max")
  - lambda_b M_b + epsilon_b) $
$ E_b^+ = op("clip")_(0,1)((1-d_E)E_b). $

In the $M_b$ update, $P_b^+$ is the value after optional budget normalization and
$M_"max"$ is positive.
$W$ is the global window modulation; $epsilon_b$ is an independent zero-mean Gaussian
draw with the branch noise scale on each consolidation pass. A zero $W$ removes
positive structural writing but leaves decay/noise active and does not by itself prevent
replay from updating $P_b$. In the current API an empty replay-ID list selects default
traces, so zero replay and zero modulation are separate conditions that must be
specified explicitly.

Recall support is computed from the activity that is already access-gated:

$ R_mu = C_mu sum_b a_(mu b) x_b, quad
  L_(mu nu) = sum_b a_(mu b) a_(nu b) M_b. $

$C_mu$ applies the context-mismatch penalty only when both current and trace contexts are
non-null and differ. `RecallSupport.support` is pre-threshold support; the separate
`expressed_strength` field applies the logistic readout. The structural $L_(mu nu)$ score
depends directly on latent $M_b$ and is an internal diagnostic, not a shared behavioral
readout across models.

== Broader continuous model variants (not the primary simulator)

The thresholded and continuous forms in this section are broader theoretical variants,
not the update rule executed in E017--E025. Their parameters are phenomenological rather
than molecule-specific. A delayed threshold $theta_delta$ can represent a coarse
write-enable condition: only when modulatory, biochemical, or replay-linked signals
exceed a local threshold does recent eligibility become a durable branch update. This
could compress saturation-like requirements on calcium/CaMKII signaling,
synaptic-capture conditions, or other plasticity-permissive gates rather than name a
directly measured concentration @Yagishita2014DopamineTiming @RedondoMorris2011STC
@Gerstner2018EligibilityTraces @Rogerson2014SynapticTaggingAllocation @Ibrahim2024STC.

The capacity term $M_"max"$ plays a different role. It is not meant to imply a hard count of microtubules, spines, or proteins. Instead, it provides a bounded phenomenological surrogate for the fact that branch-local structural support is finite: transport resources, physical occupancy, local translation capacity, and active-matter-like nonequilibrium constraints all limit how much accessibility can be durably stabilized in one compartment before turnover and competition dominate @fodor2016 @needleman2017 @Govindarajan2011DendriticBranch @Rangaraju2019MitoCompartments @Das2023LocalTranslationMemory @Hacisuleyman2024DendriticTranslation. In other words, $M_"max"$ is a reviewer-facing way to keep the model from smuggling in unlimited structural write capacity.

== Factorized Accessibility View

The central formal move in the paper is the factorization

$ A_b(t) = A_b^f(t) A_b^s(t) $

where $A_b^f$ summarizes faster dendritic, contextual, or circuit-level opening, and $A_b^s$ summarizes slower structural accessibility. This is the cleanest mathematical way to state the paper's USP: memory participation depends on a slow structural gating bias beneath faster dendritic and circuit gating.

== Continuous resource-capture extension (not the implemented update)

As a separate continuous extension, local translation or branch-specific stabilization
resources can be represented by

$ tau_P dot(P)_b = -P_b + rho_nu nu(t) + rho_"sleep" Omega(t) r_b(t) $

where $P_b$ denotes a branch-local capture or consolidation resource, $nu(t)$ denotes a global neuromodulatory or salience-related drive, $Omega(t)$ denotes consolidation-window intensity (for example spindle- or replay-rich sleep opportunity), and $r_b(t)$ denotes branch-local replay or reactivation recruitment.

and replace the slow update with

$ dot(M)_b = eta E_b(t) P_b(t) sigma(delta(t) - theta_delta) (1 - M_b(t) / M_"max") - lambda_M M_b(t) + sqrt(2 T_("eff")) xi_b(t) $

This extension is not the update used in the primary experiments. In the executable
simulator, `P_b` is a replay/sleep-driven readiness proxy; the global `modulatory_drive`
multiplies the $M_b$ write term separately. $P_b$ is not itself the access state.

== Mapping from the continuous family to the executable simulator

The previous continuous equations describe a hypothesis family. The implementation uses
fixed per-call updates with explicit clipping, a replay-derived $r_b$, the resource
state $P_b$, and a window-level $W$. A schematic Euler-style correspondence can be
written with a chosen bookkeeping interval $Delta t$ as:

$ Delta M_b approx eta E_b P_b W (1-M_b/M_"max") Delta t
  - lambda_b M_b Delta t + epsilon_b $

This relation only indicates the direction of the terms. It is not an exact derivation
of the Python update: the code uses $1-M_b/M_"max"$ rather than $M_"max"-M_b$, samples noise
per pass, clips $P_b$ and $M_b$, can normalize $P_b$ under a shared budget, and decays
$E_b$ after the update. A physical interpretation of $Delta t$ has not been specified.
The discrete equations in the previous section, rather than this approximation, define
the simulator updates analyzed in the primary result files.

== Branch Allocation And Memory Linking

An explicit branch-allocation view treats a trace as depending on a subset of branches:

$ a_(mu b) in [0, 1] $

with recall

$ R_mu(t) = C_mu sum_b a_(mu b) x_b(t) $

and a simple linking metric

$ L_(mu nu) = sum_b a_(mu b) a_(nu b) M_b(t) $

Here $x_b$ already includes effective access, so recall support is not multiplied by
$A_b$ a second time. $L_(mu nu)$ is a structural diagnostic because it contains $M_b$.
Under this view, traces with shared allocations receive a larger structural linking
score when those branches have greater $M_b$.

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

First, some consequences are largely architectural. In the executable model,
context-sensitive disambiguation can arise from the fast contextual layer, and the
structural linking score weights shared allocations multiplicatively. These outcomes
should not be treated as unique evidence for slow structural rewriting.

Second, some consequences are mechanism-specific diagnostics. Overlap-branch
$M_b$-strengthening and $M_b$-based linking necessarily depend on slow structural
writing by their definitions. They can characterize the implemented mechanism, but they
do not by themselves establish that models without $M_b$ fail a common behavioral test.

Third, the broader overlap-motif analysis showed that the theory is not confined to a single hand-built two-trace example. Linking scaled with structured overlap across weak, chain, strong, and hub motifs, while weak overlap marked a boundary condition below which a shared branch does not automatically become a structural hub. This is useful for the present theory because it suggests that the prediction is gradated: shared structural allocation should matter in proportion to how strongly replay and allocation repeatedly recruit the same dendritic subunits.

The executable bridge also clarifies one formal point. Recall support is computed from
access-gated $x_b$, without a second accessibility multiplier at readout. The theoretical
accessibility factor is applied when branch activation is formed.

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
    [$R_mu = C_mu sum_b a_(mu b) x_b$], [`RecallSupport.support = sum(weight * activation)` with access-gated `x_b`], [same pre-threshold role; context penalty applies only when both context labels are present],
  ),
) <tab-s1-theory-mapping>

#bibliography("references_slow_branch_level_accessibility.bib")
