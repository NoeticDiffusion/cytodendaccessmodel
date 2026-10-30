# Revideringsplan: Slow Branch-Level Accessibility

**Mål:** ett färdigt, internt konsekvent modellmanuskript som går att lämna till *Neural Computation* för bedömning. Planen gäller den nuvarande simulatorartikeln. Analysen av G-Node/ALM-data är ett separat spår.

**Genomförandestatus 26 september 2026:** Steg 0–2, E025-körningen och manusomskrivningen i steg 3–4 är genomförda i den här arbetskopian. E024 är bevarad som historisk gemensam-probe-baslinje; E025 är den primära, lika-budgeterade tuningjämförelsen och manuskriptets aktuella evidens. Abstract, anspråksblad, teori–kod-mappning, resultat, begränsningar, appendices och reviewer-routing är uppdaterade. Den automatiserade gränsfallschecken i steg 3 återstår. Ett första följebrevsutkast finns i `project/neural_computation/neural_computation_cover_letter.md`; den lokala, SHA-256-kontrollerade arkivkandidaten är `project/neural_computation/e025_release_candidate_2026-09-26.zip`.

Före insändning återstår: köra/registrera den fullständiga automatiska test- och reproduktionsrundan, få en oberoende modellkunnig kalläsning, publicera den lokala E025-releasekandidaten som en beständig arkivversion, lägga in arkivets DOI och fylla i insändningssystemets författar-/intresseuppgifter. MIT Press anger Editorial Manager som insändningsväg och listar en ej återbetalningsbar avgift på 125 USD; kontrollera beloppet igen vid insändning eftersom priset kan ändras ([riktlinjer](https://direct.mit.edu/neco/pages/submission-guidelines), [avgiftssida](https://mitpressjournals.mit.edu/shop/fees/?issn=0899-7667)).

**Utgångsläge (25 september 2026):** PLOS-redaktören efterfrågar tydligare presentation av bidraget, variablerna och deras biologiska tolkning. Den nuvarande versionen har redan operationaliserade begrepp, simulatorspår, nio jämförelsemodeller och en avgränsad biologisk slutsats. Det som återstår är därför framför allt att säkerställa att artikelns centrala jämförelse verkligen prövar en gemensam förutsägelse och att teori, kod och text säger samma sak.

## Beslut som styr revideringen

1. **Artikeltyp:** teoretisk modell med en körbar simulator och ett tydligt modelljämförelsetest. Simulatorn är ett medel för att undersöka en mekanistisk hypotes, inte ett påstående om en biologiskt verifierad lagringsmolekyl.
2. **Huvudfråga:** ger en långsam, grenspecifik tillgänglighetsvariabel en särskiljbar förutsägelse om senare återkallningsstöd och selektiv påverkan, jämfört med rimliga alternativ under samma indata och utvärderingsprotokoll?
3. **Primär evidens:** ett *modelloberoende utfallsmått* beräknat med samma sond/probe, tidpunkt, skala och intervention för samtliga modeller. Värden som direkt innehåller `M_b` blir mekanistiska diagnostikmått, inte krav som en modell utan `M_b` måste uppfylla.
4. **Biologisk gräns:** `M_b` är en fenomenologisk latent variabel. DANDI-analyserna i S3 är explorativa följdtester. Ny ALM-analys får inte kallas direkt validering av `M_b`.
5. **Versionsdisciplin:** bevara E017–E023 och den befintliga Zenodo-posten som historiskt underlag. Lägg en ändrad jämförelse i en ny experimentfamilj och uppdatera manuskript, figurer och anspråksregister efter dess resultat.

## Kritiska fynd som planen måste lösa

- I huvudtexten definieras `L_(mu,nu) = sum_b a_(mu,b) a_(nu,b) M_b`. SIG-B och flera strukturmått följer därför direkt av den variabel som ska prövas. En modell utan `M_b` kan falla på detta mått av konstruktion. Se manuskriptets avsnitt *Minimal formalization* och `src/cytodend_accessmodel/signatures.py`.
- `experiments/exp022_hard_comparators.py` kallar B1–B7 jämförbara utfall, men B1 beräknas med `M_b`, ackumulerad Hebb-vikt, eligibility eller resurs beroende på modell. B3 använder `M_b`-baserat länkningsmått även för alternativen. För Hebb-/eligibility-/resursmodeller sätts utfallet efter skada och räddning till samma förskadevärde; B5 = 0 följer då av implementationen. Det är ett jämförelseproblem, inte ett bevis för att dessa mekanismer saknar räddningskapacitet.
- `results/e022_hard_comparators/summary/hard_comparator_claim_ledger.md` säger att fullmodellen klarar gSIG-A ∩ gSIG-B i 1/4 motiv, medan resultatfilen markerar tre gSIG-B som `nan` när olänkade par saknas och samtidigt klassar alla fyra som `full_structural_match`. Redovisningen behöver skilja *ej definierat* från *underkänt*.
- Huvudtextens formel för `R_mu` multiplicerar med `A_b` trots att simulatorn redan har lagt in åtkomst i grenaktiviteten. S1 beskriver skillnaden, men den måste lösas eller förklaras där formeln först presenteras. Huvudtextens kontinuerliga ekvationer behöver också mappas till den diskreta, klippta uppdateringen i `src/cytodend_accessmodel/simulator.py`.

## Arbetsordning och leveranser

### 0. Lås nuläget och skriv om huvudanspråket (cirka 1–2 arbetsdagar)

- [x] Registrera nuvarande manuskriptversion, figurfilernas datum, `git`-revision, experimentkonfigurationer och vilka resultatfiler som används. Ändra inte de gamla körningarna i efterhand.
- [x] Skriv ett kort anspråksblad med: **huvudfråga, primär mätstorhet, primär jämförelse, möjlig falsifiering och uttryckliga icke-anspråk**.
- [x] Skriv två meningar för abstract: (1) vad studien introducerar, (2) vilken ny insikt en rättvis jämförelse ger. Formulera dem provisoriskt tills steg 3 är klart.

**Leverans:** `project/neural_computation/claim_contract.md` med versionsankare och beslut om vad som räknas som stöd respektive motbevis. **Klart när:** en utomstående modellforskare kan skilja simulatorns konstruktion från dess prövade slutsats.

### 1. Gör teori–kod–biologi-mappningen explicit (cirka 2–4 arbetsdagar)

- [x] Skapa en tabell i huvudtexten: symbol, matematisk funktion, exakt kodfält/uppdatering, dimensionslös enhet och tidssteg, biologisk tolkning samt vad som faktiskt kan observeras. Ta med minst `x_b`, `A_b^f`, `A_b^s`, `E_b`, `P_b`, `M_b`, replay, `R_mu` och länkningsmåttet.
- [x] Välj en enda notationsnivå i huvudtexten: antingen exakt diskret uppdatering eller tydligt märkt kontinuerlig approximation. Om båda används, skriv antagandena för övergången. Beskriv klippning till `[0, M_max]` och att brus är diskret per uppdateringssteg.
- [x] Rätta/förklara dubbel åtkomstfaktor i `R_mu`; kontrollera motsvarande mått i figurtexter och S1/S2.
- [x] Definiera `modulatory_drive`, replayöverlagring, resursbudget och noll-/skadekörning i ord före ekvationstabellen. Ange att simulatorsteg inte automatiskt motsvarar minuter eller timmar i hjärnan.
- [x] Granska varje biologisk koppling som **motivering**, **operationalisering** eller **mätbar förutsägelse**. Ingen hänvisning till actin, mikrotubuli etc. ska läsas som att en av dem identifierats med `M_b`.

**Leverans:** reviderade avsnitt *Operational Definitions*, *Conceptual Framework*, *Methods* och S1/S2. **Klart när:** varje huvudekvation kan följas till en kodrad och varje parameter har en begriplig roll.

### 2. Frys en rättvis jämförelse innan nya resultat körs (cirka 2–3 arbetsdagar)

- [x] Bestäm ett primärt gemensamt utfall: exempelvis hur mycket en cue till spår `mu` förändrar återkallningsstödet för `nu` relativt en matchad kontrollcue. Mät med samma `apply_cue`/readout, samma frysta starttillstånd per probe och samma skala för alla modeller. Kontrollera att probe-anrop inte skriver ny eligibility som läcker mellan villkor.
- [x] Definiera sekundära gemensamma utfall: kontextseparation, målinriktad kontra generell grenperturbation, och återhämtning efter samma externa intervention. Separera detta från mekanismspecifik skada på `M_b`.
- [x] Ange matchningsregel för alternativen före körning: identisk topologi, cue-sekvens, tid, brusfrön och resurs-/interventionsbudget; rimlig kalibrering av startprestation och modellers fria parametrar. Visa även resultat med oförändrade publicerade parametrar som känslighetsanalys.
- [x] Bestäm hur `N/A`, nollnämnare och motiv utan olänkade par hanteras. Ett icke definierat mått ska aldrig automatiskt räknas som fel.
- [x] Skriv i förväg vilka utfall som skulle få huvudanspråket att försvagas, exempelvis att en viktmodell matchar fullmodellen på de gemensamma utfallen efter rättvis kalibrering.

**Leverans:** ett fryst protokoll i `project/neural_computation/model_comparison_protocol.md`. **Klart när:** en kollega kan implementera utvärderingen utan att se vilka resultat som gynnar modellen.

### 3. Kör ny jämförelse och revidera slutsatsen efter utfallet (cirka 5–10 arbetsdagar)

- [x] Implementera E024 som gemensam-probe-baslinje utan att skriva över E022; därefter registrerades och kördes E025 som lika-budgeterad tuninguppföljning med samma primära mätfunktion.
- [x] Implementera gemensam grenperturbation och jämförbar intervention med fördefinierad budget. Om en alternativ modell inte kan representera en mekanismspecifik intervention, märk det som `N/A`; ge den inte automatiskt värdet noll.
- [ ] Testa utvärderingskoden på kända gränsfall: ingen cue, ingen överlappning, ingen långsam skrivning, identiska modeller, noll skada, två spår utan olänkade par. Kontrollera måttens tecken, skala och nämnare.
- [x] Kör flera motiv och frysta kandidat-/parameterområden med separat utveckling, validering och låst holdout. Rapportera effektstorlekar och tekniska analysenheter; kalla dem inte biologiska replikat.
- [x] Granska alternativens bästa *gemensamma* prestation och smalna av påståendet efter E025: fullmodellen har ingen unik fördel på den registrerade readouten.

**Leverans:** kod, test, CSV, figur och resultatsammanfattning under `results/e024_*`, plus en kort beslutsnotering. **Klart när:** samma primära mått beräknas på samma sätt för varje modell och resultatet kan reproduceras från en dokumenterad körning.

### 4. Skriv om artikeln utifrån det nya resultatet (cirka 3–5 arbetsdagar)

- [x] Abstract och första två introduktionsstyckena: problem, modell, jämförelse och slutsats utan odefinierad jargong. Säg uttryckligen vad som är simulatorresultat och vad som är biologisk hypotes.
- [x] Flytta det nya gemensamma utfallstestet före tabellen över `M_b`-specifika signaturer. Använd strukturella signaturer för att förklara *varför* modellen fungerar, inte som ensam grund för att utesluta alternativ.
- [x] Ge första figuren ett tydligt lässpår: två spår, gemensam gren, snabb respons, långsam uppdatering, en prediktion som går att testa med samma probe i olika modeller.
- [x] Strama åt Results till tre frågor: **vad producerar modellen**, **vad kan alternativen också producera**, **var finns dess begränsningar**. Ta bort upprepningar av att alla fem signaturer passerar.
- [x] Uppdatera `v2_claim_ledger.md`, `v2_figure_manifest.md`, `CLAIMS_TO_EXPERIMENTS.md`, reviewer-notebook och S2. S3/DANDI ska fortsatt vara tydligt explorativt och kan kortas om det skymmer huvudbidraget.
- [x] Skriv ett kort följebrev till *Neural Computation* som anger den specifika teoretiska insikten, jämförelsens rättvisa och vad som inte påstås om biologisk validering.

**Leverans:** nytt Typst-manuskript, PDF, figurer och uppdaterad reproduktionsväg. **Klart när:** abstract, tabell, figurer och diskussion beskriver samma resultat med samma mått och avgränsning.

### 5. Slutgranskning och insändningsbeslut (cirka 2–3 arbetsdagar)

- [ ] Kör relevanta Python-test och en ren reproduktion av samtliga manuskriptfigurer i ett dokumenterat miljöläge. Stäm av siffror mot CSV och figurtexter. Detta återstår.
- [x] Visuell kontroll av nyckelsidor i manuskript och uppdaterade appendix; kontroll av aktuell insändningsväg och avgift. Full sid-för-sid-slutkontroll görs efter arkivversionen.
- [ ] Låt minst en modellkunnig läsare läsa abstract, introduktion, modelltabell och första resultatfiguren utan projektbakgrund. Be läsaren svara skriftligt på: *Vad introduceras? Vad jämförs? Vilket resultat är modelloberoende? Vad säger detta inte om biologi?* Åtgärda missförstånd innan insändning.
- [x] Gör en sista anspråkskontroll efter E025: manuskriptet hävdar inte att långsam skrivning är nödvändig eller har unik fördel; det rapporterar comparatorernas resultat och modell-/uppgiftsgränser.

**Leverans:** insändningsklar PDF, källor, reproduktionsinstruktioner och följebrev. **Klart när:** alla fyra läsarfrågorna får avsedda svar och varje siffra har en spårbar källa.

## Lean: beslut och möjlig avgränsning

**Beslut nu: nej, Lean är inte ett krav för denna revision.** Det aktuella hindret är inte att algebra saknar maskinkontroll; det är att anspråk, mätstorheter och jämförelseprotokoll behöver bli rättvisande. Ett Lean-bevis kan bekräfta ett formellt påstående *givet dess antaganden*, men kan inte etablera att `M_b` finns biologiskt, att simulatorn fångar rätt mekanism eller att Python-implementationen motsvarar Lean-modellen utan en separat översättnings-/korrekthetsanalys.

**Möjlig liten Lean-del efter steg 3:** om manuskriptets nya huvudresultat innehåller en verklig matematisk sats, skriv först sats och pappersbevis. En kandidat är den deterministiska specialmodellen `M_(t+1) = clip(M_t + q_b (1 - M_t/M_max) - lambda M_t)`, där `q_b` sammanfattar en *konstant* eligibility–resurs–modulationsprodukt. Med `M_max > 0`, `q_b >= 0`, `lambda > 0`, `0 < lambda + q_b/M_max <= 1` och giltigt startvärde kan man härleda jämvikten `M_b* = q_b/(lambda + q_b/M_max)` och monoton konvergens mot den; detta skulle precisera när större grenspecifik skrivdrivning ger större kvarstående tillgänglighet. Det bevisar inte samma sak för tidsvarierande eligibility, resursbudget, brus eller hela Python-simulatorn. Granska först att satsen tillför en insikt utöver själva uppdateringsregeln. Formalisera sedan *just den satsen* i Lean som supplement, med exakta antaganden, reproducerbar byggrutin och utan ofullständiga bevis. Avstå om satsen inte tillför en ny insikt eller om den kräver så starka antaganden att den saknar relevans för resultaten.

**Högre matematisk prioritet:** en tydlig härledning av den diskreta modellen, dimensions-/skalkontroll, väl definierade mått och nämnare, samt en analytisk förklaring till gränserna för timing, överlapp och replay. Dessa kan bedömas av en läsare och testas mot koden före eventuell formalisering.

## Separat ALM-spår

Gör endast en Stage-0-granskning parallellt: rådata kontra figurdata, identifierbara grenar genom sessioner, djurantal, försök, regel- och interventionskohorter. Om datan tillåter en förregistrerad jämförelse av grenhistorik mot fast/global historik, starta ett separat experiment. Publicera det som en egen empirisk analys endast om effekten är robust och biologiskt intressant; även då är det ett test av `M_b`-liknande historikberoende, inte en direkt observation av `M_b`.

## Minsta ordning för att bli klar

`anspråksblad → teori–kod-tabell → fryst gemensamt jämförelseprotokoll → E024-baslinje → E025 tuning/holdout → omskrivet manus → kalläsning + arkiv → insändning`.

Lean blir aktuellt först om E024 lämnar kvar ett centralt och icke-trivialt matematiskt påstående som är värt att formalisera.
