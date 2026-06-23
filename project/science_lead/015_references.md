Yes — I did a reference audit of the main article, S1–S5, and the uploaded BibTeX file. My overall assessment is:

> **The main article is reasonably well referenced in the Introduction and biological motivation, but the supplements need more local citation support, especially S2 and S5.**
> The biggest risk is not that the paper lacks references globally; it is that some biologically loaded claims are now in figure captions, parameter tables, or supplementary explanations without nearby citations.

The main article already cites dendritic computation, spine geometry, STC/eligibility, local translation, energetic support, memory linking, and dendritic engram work in the Introduction and Biological Motivation sections.  The supplements, however, often rely on those references implicitly. Reviewers may want the local citation near the claim.

# Priority A — must fix before submission

## 1. S5 needs references in almost every biological figure caption

S5 is now explicitly conceptual and useful, but it contains biologically specific claims about spine-neck geometry, actin, dynamic microtubule entry, local translation, transport readiness, mitochondrial support, impedance gating, shared branches, memory linking, and rescue logic. The appendix correctly says these are conceptual and not direct biological validation, but the figure captions currently have no local references. 

### S5.1 — Biological motivation for (M_b)

Needs references for:

* spine-neck geometry,
* actin remodeling,
* dynamic microtubule entry,
* local translation,
* transport readiness,
* mitochondrial support.

Suggested existing BibTeX keys:

```text
Tonnesen2014
Araya2014
Popovic2015
Hotulainen2010ActinSpines
Borovac2018ActinDynamicsSpines
Merriam2011Dynamic
Merriam2013
Kapitein2011_MAP2
Dent2017MicrotubulesMemory
Dent2020DynamicMicrotubulesSynapse
Govindarajan2011DendriticBranch
Hacisuleyman2024DendriticTranslation
Das2023LocalTranslationMemory
Rangaraju2019MitoCompartments
Thomas2023PostsynapticMito
Bapat2024VAP
```

Recommended caption addition:

> These candidate processes are supported by work on spine-neck compartmentalization, actin-dependent spine remodeling, dynamic microtubule entry into spines, branch-local translation, and local mitochondrial support.

## 2. S5.2 — structural impedance gating is the highest-risk citation gap

This is the most biologically sensitive supplementary figure. It currently links microtubule state, spine-neck impedance, and signal propagation. Even with the cautionary language, this needs local references and slightly softer wording. 

Suggested references:

```text
Tonnesen2014
Araya2014
Popovic2015
Zecevic2023ElectricalPropertiesSpines
Merriam2011Dynamic
Merriam2013
hu2008activity
Kapitein2011_MAP2
Dent2020DynamicMicrotubulesSynapse
```

Recommended phrasing:

> “This schematic combines evidence that spine-neck geometry regulates electrical/chemical compartmentalization with evidence that dynamic microtubules can enter active spines and interact with activity-dependent spine remodeling. It should be read as an illustrative candidate mechanism, not as a validated one-to-one model of (M_b).”

That protects you from a reviewer saying the MT–impedance link is too literal.

## 3. S5.4 — memory linking through shared branches needs engram/linking references

This figure says episodes encoded close in time can share branch segments and become linked. That is exactly where a reviewer will expect engram / memory linking / dendritic engram citations.

Suggested keys:

```text
Choucry2024MemoryLinkingIdentity
Kastellakis2023DendriticEngram
Sehgal2025ContextLinking
Zaki2025Engram
Zaki2024OfflineLinking
Guskjolen2023EngramNeurons
Frankland2005RecentRemoteMemories
```

The main article already cites memory-linking work in the Introduction, but S5.4 should carry its own local citations because it is a standalone conceptual figure. 

## 4. S2 parameter table needs local references for biological interpretations

S2 lists canonical parameters and gives biological interpretations: `eligibility_decay` “matches STC tag window phenomenologically,” `replay_gain` scales replay recruitment, `sleep_gain` relates to sleep-window extension, and so on. 

Those biological interpretations should be cited locally.

Suggested additions:

For eligibility / tagging / consolidation:

```text
RedondoMorris2011STC
Gerstner2018EligibilityTraces
Yagishita2014DopamineTiming
Rogerson2014SynapticTaggingAllocation
```

For replay / sleep / consolidation:

```text
Wang2024SleepDependentEngramReactivation
PeyracheSeibt2020Spindles
Niethard2018
Seibt2017DendriticSpindles
Yang2014SleepSpines
TononiCirelli2020SleepPlasticity
```

For local capture / translation / branch resources:

```text
Govindarajan2011DendriticBranch
Hacisuleyman2024DendriticTranslation
Das2023LocalTranslationMemory
Daskin2025LocalProteinSynthesisSynapses
Rangaraju2019MitoCompartments
```

You do **not** need literature citations for the exact numeric parameter values, because those are simulator choices. But whenever the table says “biological interpretation,” cite the biological motivation.

## 5. S1 optional attractor-energy view needs associative-memory references

S1 contains an “Optional Attractor-Energy View” with (W_{\mathrm{eff}}) and an energy-like quantity. 

If you keep this section, it should cite classical associative-memory / attractor-model literature. Your current bib seems light here. You have:

```text
Knierim2016Tracking
Podlaski2025HighCapacity
Koch1999Biophysics
```

But I would add at least one or two canonical references if they are not already in the .bib:

```text
Hopfield 1982
Hopfield 1984
Amit, Gutfreund & Sompolinsky 1985/1987
Amit 1989
```

If you do not want to expand the bibliography, consider shortening or removing the attractor-energy view. It is optional and could invite extra reviewer critique.

---

# Priority B — should fix for a polished submission

## 6. Main Introduction: current references are good, but one gap remains

The Introduction is now much better. It cites core synaptic/recurrent accounts, dendritic computation, STC/eligibility/local translation, branch-local energy/transport, and memory linking. 

The remaining gap is the sentence about existing explanations producing context separation or partial linking through:

* fast gating,
* fixed overlap geometry,
* replay without persistent structural change,
* random slow drift,
* global gain,
* weight-only learning.

This paragraph is central to why your comparator set is legitimate. It would be stronger with references to contextual routing/gain/weight-only alternatives.

Suggested keys:

```text
Podlaski2025HighCapacity
WangYang2018Routing
Keller2020ContextualModulation
Bos2025GainModulation
Knierim2016Tracking
Koch1999Biophysics
```

Recommended revised sentence:

> Existing explanations can often produce context separation or partial linking through contextual routing, gain modulation, fixed overlap geometry, or weight-based associative strengthening...

Then cite those.

## 7. Main Operational Definitions: no references required, except optionally memory linking

Definitions usually do not need citations if they are operational. But “memory linking” is a field term. You could cite:

```text
Choucry2024MemoryLinkingIdentity
Zaki2025Engram
Sehgal2025ContextLinking
```

This is optional because the Introduction already cites memory-linking work. But if a reviewer complains, this would be easy to fix.

## 8. Main Results: mostly no external references needed

The Results are simulator findings. They should mainly cite:

* figures,
* tables,
* S2,
* S4,
* code/Zenodo.

They do **not** need neuroscience citations after every result.

For example, “SIG-C is not diagnostic” is an internal simulator result, not an external claim. “No hard comparator reproduced the full profile” is also internal. The right support is Table 3 / Figure 3 / S2, not literature.

The manuscript already routes E017–E022R through Methods and S4.

## 9. Discussion: add references for speculative future biological extensions

The Discussion says similar fast/slow access separations may be relevant in prefrontal contextual control, working memory stabilization, and systems-level consolidation, and marks those extensions as speculative. That is good. But if you keep those domains, add references.

Suggested keys:

For prefrontal/contextual control:

```text
WangYang2018Routing
Keller2020ContextualModulation
Bos2025GainModulation
Basu2016LongRangeInhibitionMemory
Muller2012InhibitoryControlDendriticExcitation
Olah2025HCNGating
```

For systems consolidation / memory stabilization:

```text
Frankland2005RecentRemoteMemories
Guskjolen2023EngramNeurons
Wang2024SleepDependentEngramReactivation
TononiCirelli2020SleepPlasticity
ReyesResina2021SleepConsolidation
```

For future branch-resolved tests:

```text
Sehgal2025ContextLinking
Kastellakis2023DendriticEngram
Uytiepo2025EngramArchitecture
```

---

# Priority C — nice to fix, but not blocking

## 10. S3 is mostly well referenced, but dataset records should be cited explicitly

S3 is one of the better supplements. It has a clear firewall: the open-data results are supplementary exploratory observable bridges, not part of the primary simulator evidence ladder. It also explains null constructions for DANDI 000718, 000336, and 001710. 

The methods are cited reasonably:

* cell registration,
* NMF,
* ripple/offline event analysis,
* cross-plane coupling,
* permutation/null logic.

But each DANDI subsection should cite the actual dataset record or dataset paper locally.

Suggested keys from your bib:

```text
Zaki2024DANDI000718
Amaya2026DANDI000336
Plitt2026DANDI001710
```

Use them at the start of each dataset subsection:

> “The 000718 analysis uses the public dataset reported in ...”

This will make the S3 provenance cleaner.

## 11. S4 software table could cite software formally

S4 lists the GitHub repository, Zenodo snapshot, software environment, RRIDs, and reproducibility routing.  This is good. It may not need conventional references if RRIDs are present, but a strict methods journal may expect formal citations for major software.

Consider adding citations for:

```text
Python
NumPy
PyTorch
matplotlib
PyNWB
DANDI CLI / DANDI Archive
pytest
Typst
```

If you do not want to clutter S4, the table with versions and RRIDs is probably acceptable. But for final journal formatting, check the venue’s software citation policy.

## 12. S1 factorized-access and resource-capture sections could cite biological motivation

S1’s minimal state-space model is your own formalization, so it does not require a citation for every equation. But phrases like “write-enable condition,” “CaMKII,” “synaptic-capture conditions,” “finite local structural capacity,” and “active-matter-like constraints” should have references. S1 already cites Yagishita, Fodor, and Needleman in that context. 

I would add:

```text
RedondoMorris2011STC
Gerstner2018EligibilityTraces
Rogerson2014SynapticTaggingAllocation
Govindarajan2011DendriticBranch
Rangaraju2019MitoCompartments
```

Especially near S1.2 and S1.4.

---

# Reference audit by manuscript section

| Section                      | Current status | Needs more refs? | Suggested action                                                           |
| ---------------------------- | -------------- | ---------------: | -------------------------------------------------------------------------- |
| Abstract                     | Fine           |               No | Abstracts often avoid citations                                            |
| Author Summary               | Fine           |               No | No citation needed                                                         |
| Introduction paragraph 1     | Good           |            Maybe | Add associative-memory / context-gating references if wanted               |
| Dendrites/spines paragraph   | Good           |               No | Already cites [3]–[7]                                                      |
| Slow stabilization paragraph | Good           |            Maybe | Add branch-local protein synthesis / dendritic allocation refs             |
| Memory-linking paragraph     | Good           |            Maybe | Add Choucry/Kastellakis/Sehgal if not already                              |
| Comparator-gap paragraph     | Moderate       |              Yes | Cite routing/gain/weight-only alternatives                                 |
| Operational Definitions      | Good           |         Optional | Add memory-linking citation if desired                                     |
| Biological Motivation        | Good           |            Maybe | Add Dent/Daskin/Govindarajan if not already                                |
| Formalization                | Fine           |               No | Original model; cite S1 not external refs                                  |
| Results                      | Good           | No external refs | Support through figures/tables/S2/S4                                       |
| Discussion                   | Moderate       |              Yes | Add references for PFC/control, systems consolidation, future branch tests |
| Methods                      | Good           |            Maybe | S2/S4 crossrefs enough                                                     |
| Data/code                    | Good           |            Maybe | Ensure Zenodo/GitHub formally citable                                      |

---

# Reference audit by supplement

| Supplement | Current status        | Main missing references                                                                                |
| ---------- | --------------------- | ------------------------------------------------------------------------------------------------------ |
| S1         | Good but technical    | STC/eligibility, branch-local translation, attractor-memory references if keeping optional energy view |
| S2         | Needs local citations | Parameter biological interpretations, STC tag window, replay/sleep support                             |
| S3         | Mostly good           | Direct dataset record citations for DANDI 000718/000336/001710                                         |
| S4         | Good                  | Optional formal software citations                                                                     |
| S5         | Needs most work       | Local citations in S5.1–S5.4, especially S5.2 microtubule/spine impedance                              |

---

# Best existing BibTeX groups to use

## Dendritic computation / nonlinear branches

```text
London2005DendriticComputation
Major2013ActiveDendrites
Larkum2009Synaptic
Spruston2008PyramidalNeurons
Sjostrom2008DendriticExcitability
schiller2000nmda
polsky2004computational
```

## Spine geometry and compartmentalization

```text
Tonnesen2014
Araya2014
Popovic2015
Zecevic2023ElectricalPropertiesSpines
```

## Actin / microtubules / spine structural dynamics

```text
Hotulainen2010ActinSpines
Borovac2018ActinDynamicsSpines
Merriam2011Dynamic
Merriam2013
Kapitein2011_MAP2
hu2008activity
Dent2017MicrotubulesMemory
Dent2020DynamicMicrotubulesSynapse
Elie2015TauCoOrganizes
```

## STC, eligibility, and delayed plasticity

```text
RedondoMorris2011STC
Gerstner2018EligibilityTraces
Yagishita2014DopamineTiming
Rogerson2014SynapticTaggingAllocation
Ibrahim2024STC
```

## Local translation and branch-local consolidation resources

```text
Govindarajan2011DendriticBranch
Hacisuleyman2024DendriticTranslation
Das2023LocalTranslationMemory
Daskin2025LocalProteinSynthesisSynapses
Seibt2012ProteinSynthesisSleep
```

## Mitochondria / local energy support

```text
Rangaraju2019MitoCompartments
Thomas2023PostsynapticMito
Bapat2024VAP
ChangReynolds2006MitoTrafficking
Mironov2004MitoTransport
```

## Sleep / replay / consolidation

```text
Niethard2018
Seibt2017DendriticSpindles
PeyracheSeibt2020Spindles
Yang2014SleepSpines
Adler2021SleepFilopodia
TononiCirelli2020SleepPlasticity
ReyesResina2021SleepConsolidation
Wang2024SleepDependentEngramReactivation
```

## Memory linking / engrams

```text
Choucry2024MemoryLinkingIdentity
Kastellakis2023DendriticEngram
Sehgal2025ContextLinking
Zaki2025Engram
Uytiepo2025EngramArchitecture
Guskjolen2023EngramNeurons
Zaki2024OfflineLinking
```

## Contextual gating / routing / gain

```text
WangYang2018Routing
Keller2020ContextualModulation
Bos2025GainModulation
Basu2016LongRangeInhibitionMemory
Muller2012InhibitoryControlDendriticExcitation
Olah2025HCNGating
Podlaski2025HighCapacity
```

## DANDI / calcium-analysis methods

```text
Sheintuch2017CellRegistration
Vergara2025CaliAli
Molter2018DetectingAssemblies
Nagayama2022NMFAssemblies
Shen2022Deconvolution
NavasOlive2024RipplAI
Liu2022ECannula
Zaki2024DANDI000718
Amaya2026DANDI000336
Plitt2026DANDI001710
```

---

# What does **not** need external references

Do not over-reference these:

* exact simulator outputs,
* signature values,
* pass/fail matrices,
* robustness percentages,
* shuffled-replay audit values,
* internal experiment labels E017–E022R,
* exact parameter-grid outcomes,
* claim-ledger statements.

Those should be supported by:

* figures,
* tables,
* S2,
* S4,
* GitHub/Zenodo,
* result files.

External literature is needed for biological plausibility and methodological precedent, not for your internal simulator outputs.

---

# My recommended implementer task

Ask for:

## E028 — Reference-placement audit and citation patch

Scope:

1. Add local citations to S5.1–S5.4.
2. Add local citations to S2 parameter biological-interpretation rows.
3. Add dataset-record citations to each S3 DANDI subsection.
4. Add contextual gating / gain / weight-only references to the Introduction comparator-gap paragraph.
5. Add STC/eligibility/local-translation references to S1.2 and S1.4.
6. Add attractor-memory references to S1.8, or shorten S1.8.
7. Add optional software citations to S4 if required by target venue.
8. Ensure all new references are present in the BibTeX file.
9. Avoid adding references to internal simulator result claims unless they are code/data availability citations.
10. Run a final unused-reference and missing-reference check.

After this, the manuscript should have a much more review-resistant citation structure: external references where the biological/methodological claims need them, internal outputs where the simulator results need them.
