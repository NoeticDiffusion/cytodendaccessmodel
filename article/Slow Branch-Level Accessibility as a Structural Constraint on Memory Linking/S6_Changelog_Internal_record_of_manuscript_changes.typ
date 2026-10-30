#import "template.typ": essay-template

#show: doc => essay-template(
  short_title: [S6 Changelog],
  doc,
)

= S6 Changelog. Internal record of manuscript changes

*Prepared 26 September 2026.* This file records changes made to the unpublished
simulator manuscript and its supporting materials. It is an internal change record,
not part of the article's evidence or claims. It is listed in the article's Supporting
Information for document history only.

== 2026-10-30 inference clarification

- Restored the byte-exact plan referenced by the E025 pre-run freeze from an unreachable Git object and verified SHA-256 0d57224d7066409f01061ffe57aa843d8ab799540f2b8a1452dc3fa690a34950. The working plan now marks later notes as retrospective and explains its earlier draft-status sentence.
- Clarified that the registered paired sign-flip tests rely on sign-exchangeability at the tested null boundary; simulator model signs were not randomized. Added a reproducible, post-hoc centered topology-cluster bootstrap sensitivity with its assumptions, results, and hashes. It reproduced all testable immediate and late registered decisions; it is explicitly not part of the preregistration.
- Corrected the E025 output description to say that saved scores are aggregated per topology draw, not individual raw probe values.

== Update after E025 and journal-fit review

- Reframed the title and article as a simulator test of slow branch accessibility, not
  as a demonstrated biological structural constraint. The introduction now recognizes
  published branch-local memory-linking evidence and states the narrower remaining
  question: whether this particular slow state adds output-level value over tuned
  alternatives.
- Replaced E024 as the manuscript's primary result with E025. The abstract, author
  summary, claim map, Results, Discussion, Methods, data statement, and supporting
  information now report the equal-budget tuning and locked holdout findings.
- Reported the corrected E025 result symmetrically: the full model exceeds five
  alternatives, is practically equivalent to shuffled replay, and is exceeded by
  Hebbian weight-only and resource-only models. Eligibility-only has no qualifying tuned
  candidate and is not assigned an inferential win, loss, or equivalence.
- Added a common-probe figure regenerated from E025 holdout draw scores. Absolute model
  means and paired contrasts are shown separately; the ±0.02 margin is applied to paired
  differences. The hub-overlap sensitivity remains separate from the primary macro.
- Added a plain-language order-of-operations guide after the executable equations, and
  tightened the description of the latent-to-observable boundary, model-defined adapters,
  topology holdout, and technical rather than biological sample units.
- Added a main-text summary of the three exploratory DANDI bundles and their analyzed
  units. The text now states that these data did not tune or validate E025 and do not
  measure $M_b$.
- Updated S1, S2, S4, and S6; the claim ledger and figure manifest; and reviewer routing.
  The earlier E024 comparison remains documented as protocol-development history and its
  source outputs are unchanged.
- Updated the project claim contract and preregistration status to distinguish E025's
  completed results from the outcome-blind plan. Updated the revision checklist and
  drafted a Neural Computation cover letter that states the bounded result and leaves
  author declarations and the future archive DOI for completion before submission.
- Checked the current MIT Press submission route and fee information; the nonrefundable
  fee is listed as USD 125 and may change before submission.
- The E025 source and run outputs are present in the repository working tree but are not
  included in the earlier Zenodo release. A local release-candidate bundle containing
  the E025 runner, figure renderer, preregistration, manifests, results, and manuscript
  PDFs has been prepared; public deposit, DOI assignment, and a final DOI-bearing rebuild
  are still needed before submission.

== Earlier update: E024 comparison and manuscript structure

== Main comparison and conclusion

- Added E024 as the primary common-probe comparison across ten simulator families,
  four primary motifs, a hub-boundary condition, and 50 paired seed clusters. The
  comparison uses a shared one-hot branch probe, frozen pre/post-consolidation state,
  and sigmoid readout, with registered model-specific state adapters.
- Replaced the earlier unique-advantage framing with the result supported by E024:
  the full-model mean was $D = 0.0104874$ (95% seed-bootstrap CI
  0.0104406--0.0105282); six of eight baseline-matched alternatives met the
  prespecified practical-equivalence rule. Hebbian weight-only and resource-only
  alternatives had larger point estimates, but the frozen decision rule classified
  those comparisons as inconclusive. Eligibility-only was practically equivalent on
  this readout but failed the baseline-match criterion.
- Clarified that these results compare specified simulator implementations and do
  not establish that slow writing is required, that the broad comparator classes are
  equivalent, or that the latent state $M_b$ is biologically validated.
- Kept context separation, branch-mask effects, targeted recovery, and mechanism
  signatures as secondary or simulator-internal results. Their model-specific values
  are not presented as independent validation of $M_b$.

== Article structure and figures

- Reorganized Results around three questions: what the common probe distinguishes,
  what internal dynamics the simulator produces, and where robustness and motif
  analyses show limits.
- Moved the E024 common-probe result ahead of the mechanism-specific signature
  profile. The signatures now describe the full simulator's internal behavior and
  are not used alone to rank alternative model families.
- Reworked Figure 1 to show the two-trace/shared-branch setup, fast and slow access,
  the simulator variables, and the biological claim boundary. Added the E024 primary
  comparison as Figure 2 and updated the figure manifest and claim-to-figure map.
- Restored the symbol-to-code mapping table in the compiled main PDF and split the
  long consolidation update across readable display equations.

== Model description and interpretation

- Replaced the earlier mixed continuous/discrete presentation in the main text with
  the simulator's discrete, clipped updates and explicit code-field mappings.
- Added the main-text symbol table for activity, fast and slow access, eligibility,
  consolidation support, structural accessibility, replay overlap, recall support,
  and the structural linking diagnostic. Each entry now states its code mapping,
  scale, interpretation, and observability limit.
- Corrected the recall-support explanation so accessibility is not multiplied twice:
  branch activity $x_b$ already includes effective access.
- Clarified replay selection, the empty replay-ID default, global support-budget
  normalization, per-pass noise, clipping, and the meaning of zero modulation.
  In particular, an empty replay-ID list is not described as a replay-free pass.
- Kept biological mechanisms such as spine geometry, cytoskeletal remodeling, local
  translation, transport, and metabolic support as motivation for a phenomenological
  latent variable, not as identified implementations of that variable.

== Supporting and reproducibility materials

- Updated S1 and S2 to match the discrete simulator, the E024 protocol and adapters,
  the comparator definitions, and the narrower interpretation of the results. The
  exploratory open-data analyses remain explicitly separate from direct measurement
  of $M_b$.
- Updated the claim ledger, figure manifest, `CLAIMS_TO_EXPERIMENTS.md`, the reviewer
  notebook and its JSON mirror, `README.md`, and `RUN.md` to route claims to scripts,
  frozen protocol, saved outputs, and figures.
- Added the E024 comparison and secondary-summary scripts, figure renderer, test,
  result tables, metadata, and figure source. E017--E023 and the historical outputs
  remain preserved as prior experiment lineage.
- Recast the Neural Computation cover letter as an initial submission letter that
  describes the E024 result without referring to a journal revision.

== Status after the E024 update (historical)

At that point, the E024-era editable article and S1/S2 PDFs had been rebuilt from Typst.
The E025 update below supersedes that status for the current manuscript.

== Files and status after E025

The editable article, S1, S2, S4, and this S6 changelog were updated from their Typst
sources. The reviewer notebook and its text mirror route readers to E025 as the primary
comparison and E024 as historical protocol development. The E025 source and outputs
have a local release-candidate bundle, but still need a versioned public archive before
submission. A submission-letter draft is maintained under `project/neural_computation/`
and has not been submitted.
