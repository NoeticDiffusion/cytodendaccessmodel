# E025: preregistreringsplan för lika-budgeterad modellkalibrering

**Version:** 1.0, preregistreringsversion  
**Datum:** 2026-09-25  
**Status:** inget E025-resultat är insamlat. Parameterregistret och de utfallsblinda topologi-/probemanifesten är genererade och hashade. Runnern låser denna version och samtliga indatahashar före första modellpoäng; efter låsning kräver varje ändring en ny protokollversion och ett nytt orört holdout.  
**Föreslaget utdatautrymme:** `results/e025_common_probe_tuning/`  
**Föreslaget skript:** `experiments/exp025_tuned_common_probe.py`

Detta är en framåtblickande plan för en ny studie efter att E024-resultaten blivit kända. E024 står kvar oförändrat som den ursprungliga jämförelsen. E025 kan ge en ny, oberoende prövning på nya topologier och dataenheter, men kan inte göra E024 till en bekräftande analys i efterhand.

## Fråga och avgränsning

Pröva om en lika-budgeterad, modellfamiljespecifik parameteroptimering ger fullmodellen större gemensamt probe-utfall än rimliga jämförelsemodeller, och om eventuell skillnad finns kvar efter att kortlivat eligibility- och replaystöd klingat av.

Studien gäller simulatorutfall. Den kan varken visa att `M_b` finns biologiskt eller identifiera en minnesmolekyl. E025 testar specificerade modeller, adaptrar och uppgifter; slutsatser om hela mekanismklasser kräver ytterligare modeller.

## Varför E025 behövs

E024 använde tio familjer, men finjusterade endast `fast_gain` mot privat-cue-återkallning före konsolidering. Mekanismparametrarna förblev vid sina registrerade värden. På baseline-matchad analys var fullmodellens genomsnittliga effekt `D=0.0104874` (95% seed-bootstrapintervall `0.0104406–0.0105282`); sex av åtta baseline-matchade alternativ uppfyllde den förhandsbestämda praktiska likvärdigheten ±0.02. Hebb- och resource-only hade större punktestimat (`0.05444` respektive `0.05259`), men E024:s ensidiga beslutsregel kunde inte klassificera jämföraröverlägsenhet och lämnade jämförelserna oavgjorda. Eligibility-only missade baseline-matchningen.

E024:s kalibreringsseeds var 0–19 och evalueringsseeds 100–149. De får inte återanvändas som E025 holdout. Dessutom är canonical-, strong- och chain-topologierna i huvudsak deterministiska; nya slumpfrön ensamma ger därför inte nya strukturella testfall.

## Förhandsbestämda hypoteser

**H1, primär:** efter tuning på utvecklings- och valideringsdata har fullmodellen ett större omedelbart `D` än varje rimlig comparator på låsta holdout-topologier, och skillnaden överstiger den praktiskt relevanta marginalen 0.02 efter multiplicitetskorrigering.

**H2, sekundär:** efter nio uttryckliga decay-only/no-replay-pass kvarstår en större `D_late` i fullmodellen än i jämförelsemodellerna. Detta är ett förhandsbestämt test av persistens, inte en ersättning för H1.

Om ett jämförbart alternativ är praktiskt likvärdigt med eller bättre än fullmodellen på holdout stöder E025 inte en generell unik fördel. Ett oavgjort resultat räknas inte som stöd för någondera modellen.

## Modeller och parameteroptimering

E025 behåller E024:s tio modellfamiljer och deras gemensamma sigmoid-readout (`gain=5`, `threshold=0.3`). Samma cue-schema, inlärningssekvens, nio konsolideringspass, probe-amplitud och `D`-definition används för alla modeller. Modellernas inre tillstånd och förhandsbeskrivna adapters får skilja sig; adapters får inte bytas efter att E025-poängen setts.

Varje familj får 32 unika konfigurationer, samma 32 `fast_gain`-värden och samma antal simulatorutvärderingar per konfiguration. De första fem punkterna återger E024:s gain-grid `1, 1.5, 2, 2.5, 3` med familjens E024-parametrar i övrigt; därmed finns både E024:s nominella värde och varje E024-kalibreringsval med. De återstående 27 punkterna använder en deterministisk Halton-design över de aktiva parametrarna. Familjer med få verksamma parametrar får inte nya mekanismer eller extra adapterfrihet för att fylla ut sitt sökutrymme. Kandidatlista och hash sparas innan utfall beräknas.

Föreslagna sökintervall utgår från E024:s nominella parametrar (`structural_lr=0.18`, `replay_gain=0.80`, `structural_gain=6.0`, `eligibility_decay=0.12`, `structural_decay=0.005`, `translation_decay=0.05`, `Hebbian LR=0.18`, drift-SD `0.025`). Exakta kandidatpunkter låses i `e025_parameter_registry.csv` före körning.

| Modellfamilj | Parametrar som får optimeras | Föreslaget intervall |
|---|---|---|
| Alla familjer | `fast_gain`, `structural_gain` | 1.0–3.0; 3–9 |
| `full_model` | `structural_lr`, `replay_gain`, `structural_decay`, `eligibility_decay`, `translation_decay` | 0.06–0.30; 0.40–1.20; 0.0025–0.020; 0.06–0.24; 0.025–0.10 |
| `soma_global_gain_only`, `shuffled_replay` | Samma verksamma parametrar som fullmodellen; respektive globala eller shuffle-mekanismer är låsta | Samma intervall som fullmodellen |
| `random_slow_drift` | Drift-SD | 0.0125–0.050 |
| `hebbian_weight_only` | `hebbian_lr`, `structural_decay`, `eligibility_decay` | 0.06–0.36; 0.0025–0.020; 0.06–0.24; E024-variantens oklippta viktuppdatering behålls |
| `eligibility_only` | `structural_decay`, `eligibility_decay` | 0.0025–0.020; 0.06–0.24; adapter `h_b=1+E_b` behålls |
| `resource_only` | `structural_decay`, `replay_gain`, `translation_decay` | 0.0025–0.020; 0.40–1.20; 0.025–0.10; adapter `h_b=1+P_b` behålls |
| `replay_no_structure` | `structural_decay` | 0.0025–0.020; strukturell skrivning förblir avstängd |
| `fast_context_only`, `fixed_allocation_only` | Inga ytterligare familjespecifika parametrar | Modellernas uppdateringar förblir avstängda |

Intervallen är förslag till E025-register, inte resultatstyrda val. E024-kodrevisionen visar att `structural_gain` och passiv `structural_decay` påverkar vissa jämförare även när strukturell skrivning är avstängd; skriptkonstanterna `hebbian_lr` och drift-SD påverkar också primärvägen. Dessa tas därför med där de är verksamma. Innan körning ska implementationen verifiera den aktiva parameterlistan per familj. Om registret behöver ändras ska skälet dokumenteras, versionen höjas och ändringen frysas innan några utvecklings- eller utfallsresultat från E025 granskas. Nya adapterkoefficienter eller viktbegränsningar skulle skapa nya modellvarianter och får inte införas tyst under befintliga namn.

## Topologier, datauppdelning och tuning

### Topologier

Varje topologidrag innehåller en topologi från vart och ett av fyra bekräftande strata (`canonical`, `strong_overlap`, `chain_overlap`, `sparse_random`) samt en separat `hub_overlap`-känslighetstopologi. Generatorns exakta versionerade regler är:

- Alla vikter som inte anges som privata eller delade sätts till 0.05. Varje privat vikt dras oberoende likformigt från [0.85, 0.95]. Alla delade vikter ligger strikt över probe-tröskeln 0.40.
- `canonical`: 4–8 grenar, två spår och en delad gren med vikter per spår från [0.70, 0.90].
- `strong_overlap`: 4–8 grenar, två spår och en delad gren med vikter per spår från [0.90, 0.98].
- `chain_overlap`: 3–4 spår; antal grenar dras likformigt mellan `2*n_traces-1` och 11. Varje spår har en egen privat gren och varje intilliggande spårpar en egen delad gren med vikter per spår från [0.70, 0.90].
- `sparse_random`: 8–16 grenar och 4–5 spår. Varje spår får en privat gren. Antal delade spårpar dras likformigt från 1 till det minsta av antalet möjliga spårpar och återstående grenar; dessa par väljs utan återläggning och vart och ett får en egen delad gren med vikter per spår från [0.55, 0.90]. Återstående grenar är neutrala.
- `hub_overlap`: 5–8 grenar och fyra spår; alla delar en hubbgren med spårspecifika vikter från [0.55, 0.85] och vart och ett har en privat gren.

Slumpströmmen för varje splitfrö och stratum härleds från SHA-256 av den fasta namnrymden `e025_topology_v1`, splitnamnet, fröet och stratumet; därmed återanvänds inte E024:s generatorström. Alla familjer och konfigurationer på samma topologi får samma dynamiska RNG-startvärde, härlett från SHA-256 av `e025_dynamics_v1|partition|draw_id|stratum`, för parade stokastiska jämförelser. Varje drag måste innehålla minst ett riktat par med både delad gren och källprivat kontrollgren i vart och ett av de fyra primära strata. Generatorn verifierar detta från allokeringarna innan körning och avbryter med fel om ett drag inte uppfyller kravet; ingen topologi väljs eller ersätts utifrån modellresultat. Fullständiga allokerings- och probemanifest sparas i `e025_topology_manifest.csv` och `e025_probe_manifest.csv` före körning.

De genererade filernas SHA-256-ankare finns bredvid varje fil i respektive `.sha256`-fil. De nuvarande värdena är: parameterregister `056c47f298e6ced335599e220bf9e47dd5b8f5f3fdbe0cbae5997a3d698697c7`, topologimanifest `e3b81a6f3783af6d15b9e4da0f08dab64776ba823941b12cd48eb7b046b325d4` och probemanifest `576c0a1bf99e6bcdb09e448fa4a3176398a8955ffe00cedc68969461f097f3d7`. Dessa ankare låser endast respektive genererade fil; hela E025-protokollet är fortfarande ett utkast tills runner och analyskod är frysta.

De fyra primära strata vägs lika i makroeffekten. Bootstrap-enheten är topologidraget, inte en upprepad seed på en identisk deterministisk topologi.

De frysta splitfröna är utveckling 200–239, validering 240–279 och holdout 300–399. Manifesten är genererade men inga modellutfall har beräknats. Holdout får inte användas för kandidatval och öppnas först efter att runner och analyskod har frysts. De fyra exakta E024-topologierna kan återköras som historisk jämförelse men räknas inte som oberoende holdout-generaliseringsdata.

`hub_overlap` redovisas som separat gräns-/känslighetsstratum och läggs inte in i fyrastrata-makrot. `D` kan beräknas där eftersom hubben och en källprivat gren finns; frånvaro av olänkade spårpar gör endast mått som uttryckligen kräver olänkade par odefinierade. Hubbens D får inte märkas `N/A` av det skälet.

### Utveckling, validering och låst holdout

1. **Utveckling:** 40 oberoende topologidrag per stratumpartition. Kör alla 32 förhandsbestämda konfigurationer per familj. Kandidater som bryter mot förhandsbestämda funktionskrav tas bort; välj de åtta högsta tränings-D-konfigurationerna per familj för validering. Rangordna fallande på makro-`D`; lika poäng bryts med minsta normaliserade parameteravstånd till E024-nominella punkten och därefter kandidat-ID. Om färre än åtta klarar kraven, valideras samtliga godkända.
2. **Validering:** 40 nya topologidrag. Bland utvecklingslistan jämförs kandidaterna enligt samma poäng och funktionskrav. Välj konfigurationen med högst lika-viktat `D`; lika resultat bryts först med minsta euklidiska avstånd till E024-nominella parametrar normaliserade med respektive sökintervall, därefter med kandidat-ID.
3. **Holdout:** 100 nya topologidrag. Kör en gång med den låsta valda konfigurationen för varje familj och en gång med E024-nominella inställningar för varje familj. Om en familj saknar godkänd kandidat efter utveckling eller validering markeras den som `no_qualifying_tuned_candidate`; dess nominella hållpunkt redovisas beskrivande, men inga bekräftande överlägsenhets-/likvärdighetsbeslut görs mot den familjen. Holdout-poäng får inte användas till omkalibrering, adapterändring eller val av ny primär hypotes.

Alla modeller körs på samma topologier och parade slumpströmmar. Slumpade comparatorhändelser får egna förhandsbestämda RNG-strömmar; upprepningar inom ett topologidrag aggregeras före inferens. Om en felaktig körning eller dataläcka upptäcks efter holdout-öppning klassas analysen som explorativ; en ny bekräftande version kräver nytt orört holdout-material.

### Baslinje- och funktionskrav

- Före konsolidering ska privat-cue-målet för varje kandidat ligga inom ±0.02 på readoutskalan från fullmodellens E024-nominella privat-cue-referens på samma topologi.
- Efter konsolidering får korrekt privat-cue-readout inte minska mer än 0.02 jämfört med samma kandidats pre-värde.
- För privat cue ska korrekt spår inte understiga det högsta felspårets readout (korrekt minus högsta felaktigt spår ≥0). Rapportera också alla absoluta mål-, kontroll- och felspårsvärden.
- Kandidater som inte klarar kraven redovisas och får inte användas för ett påstående om överlägsenhet. Ingen modell tas bort från tabellerna; status anges som baseline-/funktionskrav ej uppfyllt.

Kraven beräknas på utvecklings- och valideringspartitionerna var för sig. För varje kandidat aggregeras fel mot E024-referensen, post-minus-pre för korrekt privat återkallning och korrekt-minus-högsta-felaktigt över samtliga topologidrag inom vart och ett av de fyra primära strata; varje strata måste klara sin gräns. E024-referensen för varje topologi är fullmodellen med nominella parametrar och `fast_gain=2.0`, före konsolidering. På holdout redovisas samma mått utan kandidatbyte; om krav fallerar där klassas modellens jämförelse som ej funktionsduglig i den bekräftande tolkningen.

## Utfall

### Primärt utfall: omedelbart gemensamt probe-D

E025 behåller E024:s primära utfall. För riktat par `μ → ν` beräknas förändringen i målspårets sigmoid-readout för delad gren relativt källprivat cue:

\[
D_{\mu\to\nu}=
\big[p_\nu(\mathrm{shared})-p_\nu(\mathrm{private})\big]_{post}
-\big[p_\nu(\mathrm{shared})-p_\nu(\mathrm{private})\big]_{pre}.
\]

Varje probe startar från en egen kopia av rätt fryst tillstånd. Medelvärdesordningen är: cue-kombinationer inom riktat par, riktade par inom topologistratum, lika vikt mellan de fyra strata inom topologidrag, därefter lika vikt mellan oberoende topologidrag. Samma parmanifest och vikter används för alla familjer.

För comparator `m` definieras `Δ_m = D_full − D_m`. Positivt värde gynnar fullmodellen; negativt värde gynnar jämföraren. D-värden plottas separat från parade Δ-värden. Marginalen ±0.02 gäller Δ, inte en horisontell gräns på modellernas absoluta D.

### Sekundärt utfall: persistens efter decay-only washout

Kopior av post-konsolideringstillstånden genomgår nio passiv-decay-steg utan cue och med noll replay. I varje steg multipliceras eligibility och readiness med respektive passiva decayfaktor, och `M_b` genomgår sin vanliga passiva decay; ingen replay, strukturell skrivning, Hebb-viktuppdatering eller comparator-specifik random drift körs. Tom replay-ID-lista får inte användas som no-replay eftersom simulatorn tolkar den som standardurval. Mät därefter samma D med samma probe och readout. Rapportera `D_late`, `D_late−D_immediate` och alla komponenter; denna endpoint ersätter aldrig det primära D.

## Statistisk analys och beslut

- Rapportera per-modell `D`, `D_late`, seed-/topologivariation, 95% intervall, absoluta target-/kontrollnivåer och samtliga exkluderingar.
- För varje familj beräknas parade skillnader full minus comparator på samma topologidrag. Bootstrap och randomisering resamplar topologidrag som kluster; gör 10 000 bootstrapdrag och 100 000 parade randomiseringar.
- Slumptalsfrö för inferens är `250925`; samma bootstrap-indexdrag används för alla modeller och kontraster, medan randomiseringstest får deterministiska hypotesvisa underfrön.
- Behåll minsta praktiskt relevanta skillnad ±0.02 från E024. Definiera symmetriska utfall: fullmodellens överlägsenhet, jämförarens överlägsenhet, praktisk likvärdighet eller oavgjort. E024:s ensidiga beslutsbrist upprepas inte.
- För varje comparator och tidpunkt prövas tre ömsesidigt uteslutande utfall: fullmodellens överlägsenhet (`Δ>+0.02`), comparatorns överlägsenhet (`Δ<−0.02`) och praktisk likvärdighet (`−0.02<Δ<+0.02`, TOST). Oavgjort är standard om inget av dessa påståenden klarar korrigeringen. Holm justeras gemensamt över de 54 påståendena (9 comparatorer × 2 tidpunkter × 3 utfall), med familjevis α=0.05. För varje TOST jämförs dess giltiga sammansatta p-värde (`max` av de två ensidiga p-värdena).
- Unik omedelbar fördel hävdas endast om fullmodellens D har 95% intervall över noll och fullmodellens överlägsenhet (`Δ_m>+0.02`) klarar den gemensamma korrigeringen mot varje jämförbar, funktionsduglig comparator. Likvärdig eller bättre comparator innebär att generell unik fördel inte stöds. Oavgjort är inte likvärdighet och inte seger.
- Kandidatval sker endast på utvecklings-/valideringsdata. Holdout analyseras en gång. Ingen efterhandsändring av marginalsiffra, probe, readout, topologiviktning eller familjeurval.

## Förhandsbestämda tekniska och dokumentationskontroller

Före E025-körning ska följande vara sparat och hashat:

- denna planens låsta version och kodrevision;
- `e025_parameter_registry.csv` och `generate_e025_parameter_registry.py`, exakt 32 kandidatkonfigurationer per familj och budgetredovisning;
- `generate_e025_topology_manifests.py`, RNG-namespace, utvecklings-/validerings-/holdoutpartitioner samt fullständiga topologi- och probemanifest med hashfiler;
- modell-/adapterregister, beroendeversioner, exekveringskommando och förväntade utdata;
- preflight-kontroller för identiska modeller, frysta probe-kopior, signriktning, baseline-/funktionsfilter, hubbens definierade D och explicit no-replay.

Ny kod och data går endast till `experiments/exp025_tuned_common_probe.py` och `results/e025_common_probe_tuning/`. E024-protokollet, E024-skriptet och `results/e024_common_probe_comparison/` förblir orörda.

Innan resultattext eller figur uppdateras ska även två tidigare granskningspunkter åtgärdas i dokumentationen utan att ändra E024-resultat: (1) förklara hubbens E024-status som designexkludering, inte som matematiskt odefinierat D; (2) visa ±0.02-marginalen på parade skillnader, inte på absoluta D-värden.

## Förväntad tolkning och stoppregel

Tuning kan öka vissa modellers D men ger inte automatiskt en unik fullmodellseffekt. Eftersom E024:s Hebb- och resource-only-varianter hade större D än fullmodellen är det inte rimligt att förvänta sig att enbart finjustering av fullmodellen löser jämförelsen. Den avgörande evidensen är den låsta holdoutjämförelsen efter lika-budgeterad tuning och funktionskrav.

Om H1 inte stöds rapporteras det som avsaknad av demonstrerad unik fördel på den definierade uppgiften, inte som bevis för att `M_b` saknar all funktion. Om `D_late` visar en korrigerad och reproducerbar fullmodellfördel kan slutsatsen begränsas till persistensutfallet; den får inte skrivas om till generell nödvändighet. Alla misslyckade baseline-matchningar, tekniska `N/A`, kandidatbudgetar och negativa resultat publiceras tillsammans med positiva resultat.
