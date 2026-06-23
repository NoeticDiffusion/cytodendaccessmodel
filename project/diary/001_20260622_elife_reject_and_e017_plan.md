# 001 — 2026-06-22 — eLife desk reject, post-mortem, och plan för e017

## Session goal

Ta emot eLife desk reject, analysera kritiken rakt, och koka ner nästa steg till
ett konkret första experiment: **e017 — Traceable Simulator Core**.

---

## Vad som hände

### eLife desk reject

Manuskriptet "A branch-resolved cytoskeletal-dendritic accessibility model of associative
memory" fick ett desk reject av senior editor Panayiota Poirazi med fyra
kärnkritiker:

1. **Conceptual framing weak** — kopplingen mellan simulator och strukturell hypotes
   är för svagt utvecklad; "memory linking" och "simpler comparator baselines" odefinierade
   i abstract.
2. **Empirical evidence unconvincing** — låga effektstorlekar, dataset otillräckligt
   karakteriserade, inga alternativa dataset testade för robusthet.
3. **No simulation traces or experiment figures** — metodartikel utan visuell simulator-
   dynamik saknar rigor-grund.
4. **Technical derivations problematic and hard to follow** — ekvationerna upplevs som
   tunga utan tillräcklig pedagogisk brygga.

Citat som är viktigt att ha med sig:
> "It is likely that a number of datasets in the DANDI repository show marginal support
> for the hypothesis as encoded in an abstracted simulator. However such a finding does not
> meet eLife's standards for impact or evidential rigour."

Det är inte ett "nonsens"-avslag. Idén överlever. Manuskriptformen gör det inte i nuvarande skick.

---

## Post-mortem: vad gick fel

Artikeln försökte bära tre bevislinjer parallellt:

1. En biologisk hypotes om långsam dendritisk tillgänglighet
2. En minimal simulator
3. Tre indirekta DANDI-bryggor

Resultatet är att ingen av de tre delarna fick tillräcklig djupgående behandling.
Simulatorn presenterades mestadels som sammanfattande tabeller, inte som inspekterbar
tidsdynamik. DANDI-delarna är ärligt disclaimade som "observable bridges", men för
en editor läses disclaimers inte som bevis — de läses som en bekräftelse på att
bevisen är indirekta.

### Epistemic status på centrala claims (reviderat)

| Claim | Status |
|-------|--------|
| En minimal branch-access-simulator kan separera slow structural writing från fast gating | **Internal validated result** — men inte visuellt exponerat |
| DANDI 000718/000336/001710 stödjer hypotesen | **Exploratory bridge** — för svagt och indirekt för att bära artikeln |
| Cytoskeletal variables = minneskod | **Speculative** — aldrig påstått i artikel men kan ha lästs in |
| Simulatorn är robust över parametrar | **Plausible interpretation** — sweeps finns men ej systematiska nog |

---

## Strategiska beslut

### 1. Artikeln byggs om som simulator-first model-discrimination paper

Ny primär claim:
> A minimal branch-accessibility simulator can distinguish slow replay-dependent structural
> writing from fast contextual gating, static branch allocation, random drift, and generic
> routing in associative memory linking.

DANDI rör sig till "Exploratory observable bridges" — sektion 5 av 5, inte huvudbevisen.

### 2. Titeln förenklas

Preliminär ny titel:
> *A minimal branch-accessibility simulator for testing slow structural contributions
> to associative memory linking*

Undvika "cytoskeletal-dendritic" i huvudtiteln tills biologikopplingen är direkt och klar.

### 3. eLife överklagas inte

Kritiken är för konkret och för rimlig. Revidera inte denna version.
Frys v1 som `article/peer_review/v1_elife_submitted/`.
Skicka framtida v2 till mer specialiserad journal (computational neuroscience,
dendritic computation, eller Review Commons / PCI som eLife föreslog).

---

## Fasindelad plan för v2

Planen är dokumenterad i `article/peer_review/elife_reorganisation_plan.md`.
Komprimerat:

| Fas | Innehåll | Status |
|-----|----------|--------|
| **Phase 1 — Reframe** | Ny titel, ny claim, Operational Definitions, ny abstract | Planerad |
| **Phase 2 — Simulator visibility** | Exportera traces, generera figurer | **Nästa — e017** |
| **Phase 3 — Robustness** | One-at-a-time sweeps, heatmaps, random sampling, scaling | Väntar på Phase 2 |
| **Phase 4 — Comparators** | Utöka från 5 till 10 baselines, joint signature matrix | Väntar |
| **Phase 5 — DANDI discipline** | Dataset triage-tabell, flytta detaljer till S3 | Väntar |
| **Phase 6 — Technical simplification** | Förenkla ekvationer i huvudtext, flytta derivationer | Väntar |

---

## Nästa konkreta steg: e017 — Traceable Simulator Core

Specen finns i `project/science_lead/001_simulator.md`.

### Vad e017 är

Inget nytt vetenskapligt innehåll. Inget DANDI. Ingen ny biologi.
Enda syftet: **gör simulatorn inspekterbar som tidsserie**.

### Varför detta är rätt första steg

eLifes hårdaste och mest fixbara kritik var:
> "The absence of simulation traces or experimental figures is a major limitation
> for a methods-oriented contribution."

Det finns ingen anledning att förbättra abstract, DANDI, eller robustness innan
simulatorn producerar reviewer-läsbar dynamik. Allt annat bygger på detta.

### Vad e017 ska leverera

**Trace-export** för varje fas och varje branch:

```
init → encode_mu_1 → encode_mu_2 → pre_consolidation_probe
     → consolidation_replay → post_consolidation_probe
     → overlap_damage → post_damage_probe
     → targeted_rescue → post_rescue_probe
```

**Variabler** (per branch, per fas):
`x_b, fast_access, slow_access, effective_access, eligibility, translation_readiness, structural_accessibility`

**Recall/linking** (per trace, per fas):
`recall_support, readout_value, context_label, linking_score, overlap_contribution`

**Fem skyddade signaturer** beräknade från trace-filerna:

| Signatur | Formel |
|----------|--------|
| SIG-A | `ΔM_overlap − mean(ΔM_nonoverlap)` |
| SIG-B | `L_post − L_pre` |
| SIG-C | `R_correct_ctx − R_wrong_ctx` |
| SIG-D | `Δlinking_drop − Δrecall_drop` (under damage) |
| SIG-E | `L_targeted_rescue − L_generic_rescue` |

**Fyra figurer**:
- `Fig_e017_01_branch_state_traces.png` — fast variabler x_b, A_f, A_eff
- `Fig_e017_02_structural_accessibility_traces.png` — M_b, E_b, P_b
- `Fig_e017_03_recall_and_linking_traces.png` — R_μ1, R_μ2, L_μ1μ2
- `Fig_e017_04_signature_barplot.png` — SIG-A–E kompakt

**Utdata**:
```
results/e017_traceable_simulator_core/traces/branch_traces.csv
results/e017_traceable_simulator_core/traces/trace_support.csv
results/e017_traceable_simulator_core/traces/linking_trace.csv
results/e017_traceable_simulator_core/summary/signature_summary.csv
results/e017_traceable_simulator_core/summary/run_metadata.json
results/e017_traceable_simulator_core/README.md
results/e017_traceable_simulator_core/qc_report.md
results/e017_traceable_simulator_core/effect_summary.md
results/e017_traceable_simulator_core/claim_ledger.md
results/e017_traceable_simulator_core/figure_manifest.md
```

### Vad e017 INTE ska göra

- Inga nya DANDI-analyser
- Inga nya biologiska tolkningar
- Inga parameter sweeps
- Ingen skalning bortom fyra brancher
- Inga NDT/qMRI-kopplingar

---

## Claim ledger — session 001

| Claim | Status | Evidens | Begränsning | Nästa åtgärd |
|-------|--------|---------|-------------|--------------|
| Simulator separerar slow writing från fast gating | Internal validated result | exp001, exp013, exp014, exp015 körbara | Ej visuellt exponerat; tabellform utan dynamik | e017: trace export |
| SIG-A (overlap-branch writing) | Internal validated result | Table 2 i manuskriptet, ΔM_b1 ≈ +0.218 | Ej visuellt; parameterrobusthet okänd | e017 → e018 |
| SIG-B (linking gain) | Internal validated result | +43.1% linking i exp001 | Ej visuellt; ej robusthetstestat | e017 → e018 |
| SIG-C (context separation) | Plausible interpretation | Nämnt i manuskript; ej systematiskt testat | Saknas i canonical trace export | e017 (dokumentera om saknas) |
| SIG-D (linking mer fragilt än recall under damage) | Plausible interpretation | Nämnt i manuskript | Ej direkt beräknat från traces | e017 |
| SIG-E (targeted rescue selectivity) | Plausible interpretation | Nämnt i manuskript | Ej direkt beräknat; generic rescue saknas | e017 (partiell impl. acceptabel) |
| DANDI-data validerar cytoskeletal model | **Rejected / unsafe** | Dataset mäter offline reactivation, inte M_b | Stark burst-baseline 000718; channel sensitivity 001710 | Behåll som exploratory bridge |
| Robust parameterkänslighet | Method-limited | Sweeps i S2 nämns men ej systematiska figurer | Ej block-A–F körda | e019 (efter e017) |

---

## Filer berörda denna session

- `article/peer_review/elife_peer_review.md` — läst
- `article/peer_review/llm_response_elife_peer_review_reject.md` — läst
- `article/peer_review/elife_reorganisation_plan.md` — läst
- `project/science_lead/001_simulator.md` — läst, identifierat som nästa konkreta åtgärd
- `project/diary/001_20260622_elife_reject_and_e017_plan.md` — detta dokument

---

## Nästa rekommenderade steg

**Implementera e017 — Traceable Simulator Core.**

Konkret: skriv ett nytt experiment-script `experiments/exp017_traceable_simulator_core.py`
som wrappar den befintliga simulatorn och exporterar full tidsserie-data till
`results/e017_traceable_simulator_core/`.

Börja med att läsa `experiments/exp001_minimal_branch_linking.py` och
`experiments/exp015_comparator_baselines.py` för att förstå exakt vad
som redan finns, innan något skrivs om.

Starta **inte** e018 (comparator traces) förrän e017 producerar stabila,
inspekterbara traces.
