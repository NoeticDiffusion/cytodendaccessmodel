# Anspråksblad: Testing Slow Branch-Level Accessibility in a Simulator of Memory Linking

**Status (2026-09-26):** E024-baslinjen och den förhandsplanerade E025-tuninguppföljningen är genomförda. E025 är nu manuskriptets primära jämförelse. Dokumentet nedan bevarar E024-kontraktet och versionsankarna som historisk baslinje.

**Ursprungligt datum:** 2026-09-25

## Aktuell E025-slutsats

E025 gav varje modellfamilj 32 kandidater med samma budget, använde 40 utvecklingsdragningar och 40 separata valideringsdragningar för urval och öppnade därefter ett låst holdout med 100 topologidragningar. Fyra primära motivstrata viktades lika inom varje dragning. Resultatet gäller specificerade simulatoradaptrar, sökutrymmen, uppgiften och readouten.

- Fullmodellens omedelbara medel var `D=0.022703` (95% bootstrapintervall `0.022538–0.022864`).
- Hebbian weight-only och resource-only gav `D=0.093304` respektive `D=0.094349`; båda överträffade fullmodellen i de parade Holm-korrigerade jämförelserna.
- Fullmodellen överträffade fem kvalificerade alternativ och var praktiskt likvärdig med shuffled replay. Nio modellfamiljer fick en kvalificerad kandidat; eligibility-only fick ingen och har därför ingen bekräftande jämförelse.
- Hebbian weight-only och resource-only överträffade också fullmodellen på det sekundära `D_late`-utfallet efter nio passiv-decay-steg.

E025 stöder ingen unik utfallsfördel för fullmodellen eller ett generellt nödvändighetsanspråk för långsam grenvis skrivning. Det utesluter inte en funktion hos `M_b` i andra uppgifter, adaptrar eller modeller. Den aktuella nyheten avgränsas till den explicit körbara fenomenologiska modellen och vad denna rättvist budgeterade jämförelse visar och inte visar; principen om grenlokal plasticitet i minneslänkning presenteras som tidigare bakgrund.

## Huvudfråga

Ger en långsam, grenspecifik tillgänglighetsvariabel en särskiljbar förutsägelse om senare återkallningsstöd och selektiv påverkan, jämfört med rimliga alternativa mekanismer när samtliga modeller får samma indata och utvärderingsprotokoll?

## Vad modellen introducerar

Simulatorn innehåller en fenomenologisk grenvariabel `M_b`. Eligibility, replay-/konsolideringsstöd och en modulerande drivning uppdaterar den; dess värde påverkar senare långsam åtkomst till grenen. Variabeln är en beräkningsmässig representation av långsam, lokal tillgänglighet. Den identifierar inte en lagringsmolekyl eller biologisk process.

## Historiskt E024-kontrakt: primärt utfall och jämförelse

Det frysta E024-utfallet är post-minus-pre-förändringen i målspårets readout för en one-hot cue till en delad gren, relativt en matchad cue till källspårets privata gren. Samma cue, probe-schema, sigmoida utdatafunktion, skala och parade seeds används för alla modeller. Tillstånd-till-readout-adaptrarna är modellunika enligt det förhandsfrysta registret, så resultatet jämför dessa tio simulatorimplementationer och inte alla tänkbara modeller i varje bred mekanismfamilj.

Den primära jämförelsen ställde fullmodellen mot nio alternativ med associativ vikt, eligibility, resursstöd, replayvariation eller global gain. Topologi, cue-sekvens, uppdateringssteg och evalueringsseed parade över modellfamiljerna. Endast global input-gain kalibrerades mot privat-cue-baslinjen; en andra analys behöll publicerade parametrar. Modellurval, probe-anrop, beräkning och beslutsregel är frysta i `model_comparison_protocol.md`.

Mått som direkt innehåller `M_b` — exempelvis `L_(mu,nu) = sum_b a_(mu,b) a_(nu,b) M_b` och mått på grenvis `M_b`-förändring — är mekanistiska diagnostikmått. De kan visa hur fullmodellens interna mekanism beter sig. De får inte ensamma avgöra om en modell utan `M_b` reproducerar ett gemensamt beteendeutfall.

## Historisk E024-regel för stöd, motbevis och odefinierade mått

**Stöd för en särskiljande förutsägelse** krävde enligt protokollet att fullmodellens primäreffekt hade ett 95-procentigt intervall över noll och att dess parade skillnad mot en comparator översteg +0.02 efter Holm-korrigering. Inget av de nio alternativen uppfyllde den fullmodellsgynnande regeln. Ett pass på `M_b`-diagnostik räcker inte.

**Utfall E024:** fullmodellens primärmedel var `D=0.010487` (95% seed-bootstrap CI `0.010441–0.010528`). Sex av åtta comparatorer som klarade baslinjematchningen klassades som praktiskt likvärdiga enligt Holm-korrigerad regel; de två övriga var oavgjorda enligt den frysta asymmetriska beslutsregeln. Hebbian- och resursmodellerna hade högre medelvärden (`0.054435` respektive `0.052594`); de parade intervallen för `D_full − D_comparator` låg under `−0.02`, men protokollet testade inte formell comparatoröverlägsenhet. Eligibility-only klassades som praktiskt likvärdig men misslyckades baslinjematchningen, vilket gör dess resultat explorativt. Samma beslut uppstod vid publicerade parametrar. Utfallet försvagar därför en nödvändighets- eller unik-fördelstolkning; det etablerar inte bred ekvivalens mellan alla modeller.

Ett mått utan giltig nämnare eller utan relevanta par rapporteras som **ej definierat/N/A**. Det räknas varken som godkänt eller underkänt. Resultatfiler och sammanfattningar måste behålla den skillnaden.

## Uttryckliga icke-anspråk

- `M_b` är inte en direkt observation, unik molekylär lagringskod eller identifiering av actin, mikrotubuli, translation eller annan specifik process.
- Simulatorresultat visar inte att långsam grenåtkomst är biologiskt nödvändig eller att modellen är biologiskt validerad. E024 visar inte heller någon unik fullmodellfördel på den angivna primära uppgiften.
- DANDI-/ALM-material i S3 är explorativt. En korrelation med gren- eller historikberoende vore inte en direkt mätning eller validering av `M_b`.
- E022:s strukturella mått använder `M_b`; vissa beteendemått och skade-/räddningsberäkningar har jämförbarhetsproblem. E024 avhjälper den primära readoutfrågan för sina specificerade adaptrar, men inte för alla möjliga implementationer av dessa mekanismfamiljer.
- Ett simuleringssteg har ingen automatisk motsvarighet till en viss tidsenhet i hjärnan.

## E024-förslag till abstraktmeningar (ersatta av E025-resultatet)

1. *We introduce a branch-resolved simulator in which a phenomenological slow accessibility state, `M_b`, changes later branch recruitment after eligibility- and replay-dependent updates.*
2. *In E024, the full model showed no unique advantage on the common primary readout: six of eight baseline-matched alternatives were practically equivalent, while Hebbian and resource-only alternatives had larger point estimates and remained inconclusive under the prespecified rule.*

Resultatet gäller E024:s fyra motiv, tio definierade simulatorfamiljer, valda adaptrar och readout; det är inte en empirisk sannolikhet eller biologisk validering.

## Versionsankare och nulägesregister

Revisionens baslinje låses mot följande Git-tillstånd och filer. SHA-256-värden gör att ändrade lokala artefakter kan skiljas från denna baslinje.

| Artefakt | Nuläge vid 2026-09-25 | SHA-256 |
|---|---|---|
| Git-commit | `ce564934591f54e70d679abfa00ccddb5165c3be` (2026-07-08; `Refactor article formatting and enhance content clarity in "Slow Branch-Level Accessibility" paper. Updated title layout, author details, and abstract structure. Removed obsolete Typst source file. Changed article typ name`) | — |
| Manuskriptkälla | `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/Slow Branch-Level Accessibility as a Structural Constraint.typ`; senast ändrad 2026-07-05 | `9c05d03c56c432f3fc24d76c6be2159dbf005521ab4040f81fbe44b4293f420d` |
| Manuskript-PDF | `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/Slow Branch-Level Accessibility as a Structural Constraint.pdf`; senast ändrad 2026-07-05 | `163150cc12778db83b3d58ee47f71a2b9a885b8ff3d8d34544ed8bb1293dc2d3` |
| Figur 1 | `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/figures2/fig1_slow-branch-level-accessability-as-a-structural-constraint-on-memory-linking.png`; 2026-06-22 18:31 | `0495d8cf149136df2dd55f4bb3f8165400186b41eaaac600a36203471239b8ca` |
| Figur 2, canonical traces | `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/figures2/Fig_e023_02_canonical_traces.png`; 2026-06-23 09:24 | `af4eff1569c544239ea60ebedec989cd9c077146ff191b6f2554d9791a9c5075` |
| Figur 3, comparatorer | `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/figures2/Fig_e023_03_comparator_matrix.png`; 2026-06-23 09:24 | `7d71d88796dde2d7a4301d6cd3d69d1a2fddbba42fa927aea57027fa2ce43b84` |
| Figur 4, robusthet | `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/figures2/Fig_e023_04_robustness_landscape.png`; 2026-06-23 09:24 | `18a0f435354069bd3d180fb94808b6416f335aeb99802b0414a5c85e1623d4bb` |
| Figur 5, motiv/skala | `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/figures2/Fig_e023_05_motif_scaling.png`; 2026-06-23 09:24 | `fc886c9a64ec55003c00599537738cc6a47853bcafa9996b6ea15caa99553454` |
| Figur 6, blandad replay | `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/figures2/Fig_e023_06_shuffled_replay_audit.png`; 2026-06-23 09:24 | `07ada919435a61e1b62f2c0b2013ebf848f7f2375890c25ff2c8649e1d346677` |
| E022-kod | `experiments/exp022_hard_comparators.py`; konfigurationen är huvudsakligen inbäddad i skriptet | `18d4176b359a4914fc1bdd0973da086ecc39e35d1fa4114a867dc9aedb2c2a0e` |
| E023-figurmontering | `experiments/exp023_article_figure_assembly.py` | `c0990aea75c326ffd4934668dbc29663c0f9d7137430764adf77e991b14d9a26` |
| E022 modellbeskrivningar/parametrar | `results/e022_hard_comparators/summary/comparator_definitions.json` | `b41a9fe614a90badcaadecb52b4af90adf3a7a257dc1a6e480a67c024960b3fa` |
| E022 struktursignaturer | `results/e022_hard_comparators/summary/hard_comparator_signature_matrix.csv` | `c0bf0d33e1c9077cf6afabdc734984d2acdacfff6011214aa8da1fc871f252c8` |
| E022 beteendematris | `results/e022_hard_comparators/summary/hard_comparator_behavioral_matrix.csv` | `c2dfc036ef5de8fb130e0e0953539442a625dc89198b3b9d2a31a5d29e3683f2` |
| E022 specificitetsmatris | `results/e022_hard_comparators/summary/hard_comparator_specificity_matrix.csv` | `a30dfa5545415965cc695eebf15bb17ac6f6b3083f2a34afd3f67803d6f7c97c` |
| E022 pass/fail-sammanfattning | `results/e022_hard_comparators/summary/failure_mode_summary.csv` | `9eb1d55bfad9f3da10c1e1dc2cac5e02d3df590d94caf0d12922c768ba851116` |
| E022 claim ledger | `results/e022_hard_comparators/summary/hard_comparator_claim_ledger.md` | `5785f4540b02dfd82990e09a5cb99476781fdcb8a7168151afd803d94f8e0a99` |

Körningsfamiljerna E017–E022R finns kvar under `results/`; E023 monterar artikelns figurer från dessa resultat. E022 har JSON-konfiguration för jämförarmodellerna, medan motiv- och körparametrar i huvudsak definieras i Python-skripten. Ingen E024-körning är del av denna baslinje.

### Konfigurations- och resultatrouting för nuvarande huvudfigurer

De äldre experimentinställningarna ligger huvudsakligen i Python-skripten; E022:s jämförarparametrar finns även i JSON-filen ovan. E023 läser följande befintliga utdata när den bygger figur 2–6. SHA-256 är med för skript och centrala indata; CSV-filerna lämnas oförändrade.

| Familj/figur | Konfigurationskälla | Resultatfiler som figuren läser | SHA-256 för skript / resultatfiler |
|---|---|---|---|
| E017 / figur 2 | `experiments/exp017_traceable_simulator_core.py` | `results/e017_traceable_simulator_core/traces/branch_traces.csv`; `results/e017_traceable_simulator_core/traces/linking_trace.csv` | script `c5b7affd29658fa19c302bdf77f64caba0e2f91a44143496535cdadd62ebb494`; branch `38affc4843a4e55e25afe2067bfe6fea19441d415e0a3c0838ee52e36f4cc911`; linking `b16d821ccd84dcda1ae7c539f8f6d4d373d390f40250cd48b3995399f01a7145` |
| E018 + E022 / figur 3 | `experiments/exp018_comparator_trace_matrix.py`; `experiments/exp022_hard_comparators.py`; E022 JSON-konfiguration ovan | `results/e018_comparator_trace_matrix/summary/comparator_pass_fail_matrix.csv`; E022 signatur- och beteendematris ovan | E018 script `1669ab877e369215745b18f682d4263339dc983a2cfcddddc2213dfa6dbd782f`; E018 matrix `d327c14a557a8b4e5c45206e59b99321d750a4104f0e4a838b5a0aabbfd2dce8`; E022 script och resultat finns i föregående tabell |
| E019 / figur 4 | `experiments/exp019_one_at_a_time_parameter_robustness.py` | `results/e019_one_at_a_time_parameter_robustness/summary/robustness_summary_by_parameter.csv` | script `0573157a6f998427a02a42313e0f5a103c5b22b58c8cd02b2868630dbacac5fa`; result `91fb5b758e0185596fdf914f72e208f44bdfe77f63699988882a3603d9d14187` |
| E019R / canonical tabell | `experiments/exp019r_signature_protocol_lock.py` | `results/e019r_signature_protocol_lock/summary/canonical_reproduction_table.csv` | script `aa6d84460216b1edcf38a6591a03491799be9f5b866d24957ce6c711e92a47f8`; result `7a093ed002608e723fc78d84a3f8f079a6db3b10e9138fdf4e90e223d381f7db` |
| E020 / figur 4 | `experiments/exp020_two_parameter_robustness_heatmaps.py` | `results/e020_two_parameter_robustness_heatmaps/summary/all_heatmaps_long.csv`; `results/e020_two_parameter_robustness_heatmaps/summary/canonical_reference.json` | script `56a0dd0e577fcb547e1f9311ffb05859822c11f7d38ea468f63fafc51b639048`; heatmaps `11ff3cb45a923f614754fe5d540e35d2338d56dc022c963710409fbb738682fc`; reference `2aa53081bba397d30b6d9bbd5cad3827d1c8a93f5479ce2d0162e99cfa24f199` |
| E021 + E021R / figur 5 | `experiments/exp021_scaling_and_motif_generalization.py`; `experiments/exp021r_generalized_specificity_gate.py` | `results/e021_scaling_and_motif_generalization/summary/generalization_summary_by_motif.csv`; `results/e021r_generalized_specificity_gate/summary/motif_specificity_classification.csv` | E021 script `2a45204a143357e4a087a713bdfdcabb1d29088f30f703d75680fa6b1b453995`; summary `dce0a31a746975333a2ce46e26cd0a5d8168286fde53085970e56ea1d3006666`; E021R script `956d6cf2f33913cd39f7399f182ab3c19a679bf118225f8fef9ae4caa890f9bb`; classification `df362db411905d3a76464150a8e956151d0b376d559b42d30b133e0c34c14e9c` |
| E022R / figur 6 | `experiments/exp022r_shuffled_replay_scaling_audit.py` | `results/e022r_shuffled_replay_scaling_audit/summary/shuffled_replay_vs_full_model.csv`; `results/e022r_shuffled_replay_scaling_audit/summary/shuffled_replay_scaling.csv` | script `05970b12034a5cc691dabffb2dec7f90c57e1dbd5db9ff2ece1166b9733fb272`; comparison `323c46f3c4427558b77289b52356a5a7fcbb8af19482b7996d8b479996b123da`; scaling `48e6afdf6dd725064145c21182159bca6cbeee2ed512ae4ecfa1b9285799348f` |
| E023 / figurmontering | `experiments/exp023_article_figure_assembly.py` | Läser filerna i raderna ovan och skapar `figures2/Fig_e023_02`–`Fig_e023_06`; konceptfigur 1 ritas i skriptet | script `c0990aea75c326ffd4934668dbc29663c0f9d7137430764adf77e991b14d9a26`; monterade PNG:er hashas i föregående tabell |

Arbetskopian hade redan otrackade kataloger under `project/` vid revisionens start, inklusive `project/neural_computation/` med revisionsplanen och bakgrundstexten. Dessa befintliga filer har lästs men inte skrivits över; detta anspråksblad är en ny fil.

## E024-resultatankare efter baslinjen

Den frysta jämförelsen finns under `results/e024_common_probe_comparison/`. Protokollet
(`model_comparison_protocol.md`) hade SHA-256
`b14403fe0a6cdf8969060c2e006459ffbdeb8a959c11ec89e3f5d58bfec62e94`; den utvärderade
huvudimplementationen `experiments/exp024_common_probe_comparison.py` hade SHA-256
`b5b745d09249fe8348854a5184f228af751fbd1d8da7b63d232a00f21281274c`, i överensstämmelse
med `metadata/run_config.json`.

| E024-artefakt | SHA-256 |
|---|---|
| `metadata/run_config.json` | `1931a3a736dffe37e05725143dd876fd81ced010a66778e499eae9daa0ec94a9` |
| `summary/primary_by_seed.csv` | `9450ac7a5139e0ec6b5f7e2d28aab18043292fc06b235738a699c69a79c98a03` |
| `summary/primary_model_summary.csv` | `28e3313a17d4d3e2258d681e1ddefbdd0321729464d9f46f38594408eb5df49f` |
| `summary/paired_comparator_summary.csv` | `d95c7dee2a5a869bc7b32fc61b232c189e63999a93399a46f3e6d27f5606781c` |
| `summary/probe_manifest.csv` | `bf93c9baecf911be9ba5c223640d95cca2db62a38e4d40a118b44e5506eacf9f` |
| `summary/claim_ledger.md` | `86b6f9f30c9d0beb1fbb0b26258c2eddde4001ff21848b80999563d11a4d104f` |
| `summary/recovery_model_summary.csv` | `fe56a9da63d9c4c686d32cdc98c94b30d3316eb312e13de34268dd76b1f3b937` |

E024's figure-language conversion and registered recovery aggregation are post-processing
only. Their scripts and input/output hashes are recorded in
`results/e024_common_probe_comparison/metadata/figure_postprocessing.md`; they do not
change the frozen primary CSVs.

## Baslinjens granskningsfynd och hantering

1. I huvudtexten och `src/cytodend_accessmodel/signatures.py` definieras strukturlänkning med `M_b`. Det är därför ett mekanismmått och får inte användas som enda utfall mot modeller som saknar `M_b`.
2. E022 använder olika beteenderepresentationer beroende på jämförarmodell. B3 bygger på strukturlänkning även för alternativ utan `M_b`; B5 kan bli noll när post-skade- och räddningsvärden avsiktligt återanvänder samma före-skadevärde. Dessa resultat visar inte att alternativen saknar räddningsförmåga.
3. E022:s fyra `full_model`-rader har gSIG-B `NaN` i tre motiv utan olänkade par, men etiketteras ändå `full_structural_match`. Claim ledger räknar samtidigt pass i 1/4 motiv. `NaN` måste redovisas som ej definierat; pass/fail måste räknas endast där måttet är definierat.
4. **Baslinjefel, korrigerat i utkastet:** den gamla formeln för `R_mu` multiplicerade med `A_b` trots att `apply_cue` redan access-gatar `activation`. Huvudtext och S1 summerar nu access-gated aktivitet utan en andra faktor.
5. **Baslinjefel, korrigerat i utkastet:** kontinuerliga ekvationer framställdes som exekverbara trots att kodens cue- och konsolideringssteg är diskreta. Huvudtexten anger nu diskreta uppdateringar, klippning, budget, uppdateringsordning och brus; S1 märker den bredare kontinuerliga familjen som en variant och anger dess begränsade Euler-koppling.
6. I simulatorn betyder `replay_trace_ids=()` inte noll replay: `_select_replay_ids` tolkar en tom lista som standardvalet av alla kontexttillåtna spår. `modulatory_drive=0` stänger den positiva skrivtermen för `M_b`, men replay kan fortfarande uppdatera `P_b`; decay (och eventuellt brus) fortsätter. E022:s så kallade nullpass är därför noll modulation, inte en replayfri konsolidering. Bevara detta som historik och gör en explicit, entydig noll-replay-representation i den nya utvärderingen.
