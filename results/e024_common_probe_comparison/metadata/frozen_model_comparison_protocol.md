# Fryst jämförelseprotokoll för E024

**Protokollversion:** 1.1  
**Låst:** 2026-09-25, före körning av E024-modelljämförelsen  
**Status:** specifikation för implementering; inga E024-resultat ingår  
**Utdata:** `results/e024_common_probe_comparison/`  
**Föreslaget skript:** `experiments/exp024_common_probe_comparison.py`

Detta protokoll anger modellurval, primärt utfall, sonder, kalibrering, sekundära utfall och beslutsregler. E017–E023 och den nuvarande Zenodo-posten är historiska resultat och ska inte skrivas om. E024 ska använda en gemensam utvärderingsväg; äldre modellunika B1–B7-värden återanvänds inte som primära E024-utfall.

## Fråga och avgränsat anspråk

Pröva om långsam, grenspecifik tillgänglighet förbättrar **förvärvat stöd för en annan minnesspårning** efter en gemensam inlärnings- och replaysekvens, jämfört med alternativa mekanismer under samma indata och probe/readout. Utfallet gäller simulatorns beteende. Det identifierar inte `M_b` med en biologisk molekyl och testar inte biologisk nödvändighet.

## Modeller

Ta med fullmodellen och samtliga publicerade comparatorfamiljer som kan implementeras bakom det gemensamma utfallsgränssnittet:

| Källa | Modellfamiljer |
|---|---|
| E018 | `full_model`, `fast_context_only`, `replay_no_structure`, `random_slow_drift`, `fixed_allocation_only` |
| E022 | `full_model`, `hebbian_weight_only`, `soma_global_gain_only`, `shuffled_replay`, `eligibility_only`, `resource_only` |

`full_model` körs en gång per konfiguration; listan ger totalt tio unika familjer. E024:s modellregister ska beskriva varje familjs tillstånd, uppdateringsregel, fria parametrar och readoutadapter. En comparator får inte använda ett utfallsmått som är konstruerat separat för just den mekanismen. Om en familj inte kan lämna samma målspårsstöd via den gemensamma readouten markeras den `N/A` med teknisk orsak; den får varken nollpoäng eller klassas som falsifierad. En sådan familj ska inte ingå i ett påstående om att fullmodellen besegrat den.

## Gemensam uppgift, readout och primärt utfall

### Gemensam readout

Varje modelladapter ska för en given tillståndskopia och cue lämna ett pre-tröskelstöd `s_j` för **varje** registrerat målspår `j`. Värdet ska ha samma semantik i alla familjer: evidens för återkallning av målspåret efter just den givna cue:n. Samma slutreadout används överallt:

\[
p_j=\sigma\!\left(5\,(s_j-0.3)\right),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

`p_j` är ett dimensionslöst readoutvärde på skalan 0–1; det är inte empiriskt kalibrerad sannolikhet. Fullmodellen använder sin vanliga cue-till-aktivitet- och trace-supportkedja. En alternativ modell får ha en mekanismspecifik intern uppdatering, men dess adapter måste beräkna samma målspårsstöd i samma uppgiftssteg. Adapterns ekvationer fryses i modellregistret innan evalueringskörningarna. Inga `M_b`-baserade länkningsmått används i primär readout.

E024 operationaliserar den gemensamma readouten med

\[
s_j=\sum_b W_{j b}\,h_b\,x_b,
\]

där `x_b` är grenaktiviteten efter samma cue, `W_jb` är modellens branch-till-spår-vikt och `h_b` är en mekanismens readoutgain (normalt 1). Modellerna använder följande frysta adaptrar:

| Modellfamilj | `W_jb` och `h_b` / dynamik |
|---|---|
| `full_model`, `fast_context_only`, `replay_no_structure`, `random_slow_drift`, `fixed_allocation_only`, `shuffled_replay` | `W_jb` är motivets ursprungliga spårallokering; `h_b=1`. Respektive gren- och replaydynamik följer E018/E022. `random_slow_drift` tillför Gaussisk `M_b`-drift med `σ=0.025` per konsolideringspass utan replayuppdatering. |
| `hebbian_weight_only` | `M_b` skrivs inte. Vid varje konsolideringspass uppdateras `W_jb ← W_jb + 0.18 E_b a_jb`, med `E_b` före passet och ursprunglig allokering `a_jb`; vikten ackumuleras utan klippning. |
| `eligibility_only` | `W_jb=a_jb`, `h_b=1+E_b`; samma replayhändelser erbjuds som övriga modeller men `replay_gain=0`, så de skapar ingen readiness. Eligibility klingar av under passen. |
| `resource_only` | `W_jb=a_jb`, `h_b=1+P_b`; replay laddar `P_b`, medan `M_b` inte skrivs. |
| `soma_global_gain_only` | `W_jb=1` för alla grenar och spår; eligibility plattas till sitt medel över grenar före konsolidering, enligt E022. |

I samtliga fall appliceras eventuell kontextmismatch med simulatorns gemensamma `context_mismatch_penalty` före sigmoiden. Denna adaptertabell och samtliga valda parametrar exporteras i modellregistret före evaluering.

### Inlärningssekvens och frysta probe-tillstånd

För varje modell, motiv och seed:

1. Initiera samma gren-/spårtopologi, spår-ID:n och yttre cue-definitioner. Modellernas interna representation kan skilja sig enligt fördeklarerad comparatorregel; initialtillstånd och initiering ska beskrivas i modellregistret.
2. Presentera varje spår i motivets fördeklarerade ordning med samma yttre allokeringscue två gånger. Använd samma cue-amplitud, kontext och tidpunkt i alla modeller. Lägg inte in probe-cues i inlärningsfasen.
3. Spara ett **pre-konsolideringstillstånd**.
4. Kör nio konsolideringspass med alla spår uttryckligen valda för replay, `modulatory_drive=1.0` och `sleep_drive=0.0`. Spara ett **post-konsolideringstillstånd**.
5. Kör varje probe från en separat djup kopia av respektive fryst tillstånd. I simulatorn anropar `apply_cue` också eligibility- och readiness-uppdateringar. Dessa mutationer får inte föras vidare till någon annan probe eller ett senare tillstånd. Ingen probe följs av konsolidering.

Den primära cue:n är en one-hot grenprobe med amplitud 1.0 på en gren som delas av källspår `μ` och målspår `ν` (allokering >0.40 till båda). Den matchade kontrollcue:n har samma amplitud och cue-längd på en källprivat gren (>0.40 till `μ`, ≤0.40 till `ν`). Använd alla sådana delade respektive källprivata grenar; medelvärdesbilda först över grenkombinationer inom ett riktat spårpar. Villkoren, cue-vektorerna och paren bestäms enbart från motivets allokeringar, före modellkörning.

### Primärt estimand

För riktat par `μ → ν`, delad gren `b` och matchad privat gren `c`, beräkna:

\[
D_{\mu\to\nu}=\left[p_\nu(b)-p_\nu(c)\right]_{post}
-\left[p_\nu(b)-p_\nu(c)\right]_{pre}.
\]

Detta är förändringen i målspårets återkallningsstöd som är specifik för en cue till delad gren jämfört med en lika stark cue till källans privata gren. Den primära E024-effekten är medelvärdet av `D` över riktade par inom varje motiv och därefter medelvärdet över motiv med lika vikt. Rapportera även varje motivs effekt separat. Rapportera både fullmodellens effekt och den parade skillnaden `D_full − D_comparator`; positiva värden gynnar större cue-selektivt förvärv i fullmodellen.

Om ett motiv saknar något riktat par med både delad och källprivat kontrollgren är dess primärutfall `N/A`. Det ersätts inte med noll. En parlista och nämnare ska sparas till CSV.

## Motiver och slumpning

Kör de fyra E022-motiven: `canonical`, `strong_overlap`, `chain_overlap` och `sparse_random`, med samma gren-/spårantal och allokeringströskel som de frysta motifbyggarna. Lägg till `hub_overlap` endast som fördeklarerat gränsfall; eftersom alla par delar hubben finns där inget olänkat par och det motivet bidrar inte till primärmedelvärdet.

Använd evalueringsseeds `100–149` för slumpade motiv och stokastiska comparatorsteg. Samma seed och motiv förekommer i varje modellfamilj (parad jämförelse). Primäranalysen körs med `structural_noise=0.0`, så den motsvarar E022:s publicerade grundläge. Om stokastisk strukturell variation studeras, redovisas den som separat känslighetsanalys för alla modeller med samma noise-nivå och seeds; den ersätter inte grundanalysen.

## Parameterhantering och rättvis matchning

### Primär, baseline-matchad analys

- Behåll varje modells definierande mekanism och dess E018/E022-nominella parametrar.
- Kalibrera endast den gemensamma `fast_gain` med samma kandidater för alla familjer: `{1.0, 1.5, 2.0, 2.5, 3.0}`. Övriga readoutkonstanter förblir `readout_gain=5.0` och `readout_threshold=0.3`.
- Referensen är fullmodellen med E022-värden och `fast_gain=2.0`. På kalibreringsmotivet `canonical`, seeds `0–19`, välj för varje familj den gain som minimerar absolut skillnad från fullmodellens **pre-konsoliderings korrekta privata-cue-återkallning**. Använd inte delad-cue, post-konsoliderings-, skade- eller primärutfall vid kalibreringen. Vid lika avstånd väljs värdet närmast 2.0. Om ingen kandidat ligger inom ±0.02 enheter på readoutskalan, använd den närmaste och märk familjen som `baseline_match_failed`; ta med den i utvärderingen och visa missmatchen.
- Frys vald gain per familj innan evalueringsseeds 100–149 körs. Spara hela kandidattabellen, kalibreringsutfall och konfigurationshash.

Den gemensamma kalibreringen är avsiktligt begränsad till en global input-gain. E024 får inte finjustera mekanismparametrar mot det primära utfallet. Som obligatorisk känslighetsanalys körs samma modeller med oförändrade publicerade E018/E022-parametrar och `fast_gain=2.0`.

## Fördeklarerade sekundära utfall

Dessa utfall redovisas separat och får inte ersätta ett oklart primärutfall:

1. **Kontextseparation:** för varje privat cue, `p_correct − max(p_wrong)` med samma readout. Medelvärdesbilda över spår; markera `N/A` om motivet har färre än två spår eller modellen saknar kontextinmatning.
2. **Gemensam grenperturbation:** upprepa samma frysta prober med en fördeklarerad grenmask som sätter bidraget från en vald gren till noll i den gemensamma readoutvägen. Jämför stödförlust när den delade grenen maskas med stödförlust när en lika många-grenar stor källprivat kontroll maskas. Detta är en output-level perturbation, inte en skada på `M_b`.
3. **Återhämtning efter perturbation:** från posttillståndet applicera en gemensam 50-procentig gainreduktion på den valda delade grenen vid både cue-inmatning och probe-readout. Mät primärutfallet under perturbationen. Jämför sedan två kopior med samma kvarvarande gainreduktion: (a) tre rundor med tre cue-repetitioner på den delade grenen per runda, där cue-gain är 1.0 före perturbationen, och (b) samma antal cues på en matchad källprivat gren med cue-amplitud 0.5. Kör tre konsolideringspass efter varje runda med identisk explicit replaylista i båda armarna. Mät primärutfallet med perturbationen kvar och därefter med gainreduktionen borttagen. Rapportera riktad minus kontrollförändring och absoluta värden; armarna har samma antal cue- och konsolideringssteg.
4. **Mekanismdiagnostik:** `M_b`, `E_b`, `P_b`, synaptiska vikter och replaybokföring får visas som separata interna förklaringar. De är inte modelloberoende beteendeutfall.

Ett eventuellt replayfritt kontrolltillstånd ska skapas med en explicit `replay_mode="none"`/motsvarande API i E024-adaptern. En tom `replay_trace_ids=[]` får inte användas: nuvarande simulator tolkar tom lista som standardurval av kontexttillåtna spår. Noll `modulatory_drive` är heller inte replayfritt, eftersom `P_b` fortfarande kan ändras.

## Saknade värden och nämnare

- `N/A` används om ett fördeklarerat utfall inte är definierat av motivet eller av modellens specificerade interface. Skriv orsakskod, exempelvis `no_eligible_pair`, `no_unlinked_pairs` eller `unsupported_readout`.
- `NaN`, tom parlista eller noll nämnare är `N/A`; de räknas inte som noll, pass eller fail.
- Samtliga comparatorer jämförs med samma paruppsättning och motivnämnare för ett utfall. Om en modell saknar värde tas den inte tyst bort från jämförelsematrisen.
- Redovisa antal definierade par, motiv och seeds tillsammans med medelvärden.

## Sammanfattning och beslutsregel

För varje seed beräknas först medelvärdet över riktade par inom motiv. Därefter tas ett lika viktat medel över de motiv där parmanifestet anger primärutfallet som definierat. Paruppsättningen är identisk mellan modeller och fastställs från allokeringarna före resultatberäkning. Rapportera motivspecifika effekter och 95-procentiga bootstrapintervall över seed-kluster med 10 000 parade resamplingar; använd samma resamplade seeds för alla modeller. Visa råa effekter; pass/fail används inte som primär evidens.

Förhandsbestämd minsta relevant skillnad är `0.02` på readoutskalan 0–1:

- **Stöd för särskiljning:** fullmodellens primäreffekt har ett 95-procentigt intervall helt över noll och dess parade skillnad mot jämföraren överstiger `+0.02` enligt den multiplicitetskorrigerade regeln nedan.
- **Praktisk likvärdighet:** den parade skillnaden är inom `[-0.02, +0.02]` enligt det multiplicitetskorrigerade likvärdighetstestet nedan. Då stöder jämförelsen inte att fullmodellen behövs för detta utfall.
- **Oavgjort:** intervallet korsar gränserna för stöd och likvärdighet, eller datat är för glest/utfallet N/A. Dra då ingen diskriminerande slutsats.

Använd parade randomiseringstest över seed-kluster med 100 000 teckenbyten och fast testseed `240925`. För särskiljning testas ensidigt om den genomsnittliga skillnaden överstiger `0.02`; för likvärdighet används två ensidiga test mot gränserna `−0.02` och `+0.02`. Holm-korrigera p-värdena med familjevis α=0.05 över de nio fullmodell–comparatorjämförelserna (respektive alla 18 ensidiga test för likvärdighet). Rapportera bootstrapintervall och korrigerade beslut tillsammans. Motivspecifika resultat är beskrivande om inte en separat multiplicitetsregel fördeklareras.

Sammanfatta modellerna utifrån det bästa jämförbara alternativet på det gemensamma utfallet. Att ett alternativ saknar `M_b` eller fallerar på `M_b`-specifika signaturer räknas inte som nederlag i den primära jämförelsen. Om någon rimligt baseline-matchad comparator är praktiskt likvärdig på primärutfallet ska manuskriptets anspråk begränsas till de mekanistiska skillnader eller motiv där data faktiskt skiljer modellerna.

## Körningsspärr och reproducerbarhet

Före första E024-evalueringskörningen ska följande finnas under `results/e024_common_probe_comparison/`:

- kopia av detta protokoll med SHA-256;
- modellregister och konfiguration med källhashar, parameteruppsättningar och vald kalibrering;
- probe-/parmanifest inklusive N/A-status före resultataggregering;
- versionsinformation för kod och miljö.

E024 ska spara råa probevärden och tillstånds-ID:n så att varje sammanfattat värde kan räknas om. Befintliga `results/e017_*`–`results/e023_*` lämnas orörda. Det finns redan `experiments/exp024_concept_figure_pack.py`; jämförelseskriptet ska därför få ett separat namn och får inte skriva över den filen eller dess utdata.

## Före körning: tekniska kontrollpunkter

Det här är implementationskontroller, inte empiriska pass/fail-kriterier:

- varje adapter returnerar stöd för samma målspår med samma readout, även när interna tillstånd skiljer sig;
- alla prober börjar från oberoende kopior av rätt fryst tillstånd;
- identiska modeller ger `D_full − D_comparator = 0` numeriskt;
- i ett särskilt nolltillstånd utan cue ger adaptern `s_j=0` före readout (kontrollera sigmoidens 0.182-baslinje separat); testa gren-cues mot nollallokering i ett separat, omodifierat fixture;
- N/A-fall (inklusive inga giltiga delade/privata par och motiv utan olänkade par) förblir N/A i CSV och figurer;
- en explicit replayfri kontroll rapporterar noll valda replay-spår och ändrar inte `P_b` genom replayöverlapp;
- jämförarens parameterkalibrering läser enbart den tillåtna pre-konsolideringsbaslinjen.

Kör inte den frysta evalueringsmatrisen förrän dessa kontroller passerar och modellregistret/probe-manifestet har sparats. Då kan E024-körningen börja utan att se eller anpassa sig till dess primärresultat.
