# Published data for calibrating and validating the maxillary FE + remodeling model

Prepared 2026-10-17 by the evidence researcher. Full table: `data_sources.csv` (44 rows, IDs below). Status flags: values marked "read" were taken from a fetched page; anything else is **UNVERIFIED**. Authors were not always visible in the pages I read, so some citations give titles and URLs only. Verify before submission.

## Key findings

1. **No source measures strain on the human adolescent maxilla in vivo.** The nearest data are adult cadaver loading, pig in vivo chewing, and macaque in vivo chewing.
2. **No source tests whether tongue posture changes maxillary bone density.** The only human posture-therapy data I found are tongue-pressure and posture outcomes (C8, D12), not bone.
3. **The reference resting pressure in `configs/study.yaml` (200 Pa) matches the only direct measurement I found.** That measurement is 106-254 Pa on a lower-arch tongue pad in 19 adults ([PubMed 26679665](https://pubmed.ncbi.nlm.nih.gov/26679665/)). I found no measurement for a deliberate 2 kPa posture, so the mewing scenario stays an assumption.
4. **Swallow peak pressure in the config (15 kPa) is below adult IOPI values.** Adults aged 20-29 reached about 50 kPa for saliva swallows ([Lin 2021](https://www.mdpi.com/1660-4601/18/10/5452)). Children 6-12 reached 18.5-28.2 kPa ([PMC8869793](https://pmc.ncbi.nlm.nih.gov/articles/PMC8869793/)). The sweep range (5-30 kPa) covers children but not adults. The IOPI bulb is a different contact geometry from a distributed palatal load, so the comparison is approximate.
5. **Chew force of 100 N is plausible but only loosely supported.** The pooled maximum bite force of permanent dentition is 489 N ([PubMed 37143420](https://pubmed.ncbi.nlm.nih.gov/37143420/)). Adolescent anterior maximum is about 170-220 N ([Verkouteren 2022](https://onlinelibrary.wiley.com/doi/full/10.1111/eos.12869)). Habitual chewing force per cycle was not found, so it stays an assumption.

## A. Ex vivo strain on maxilla/palate

| ID | Study | Load | Sites | Result |
|---|---|---|---|---|
| A1/A2 | Pig skull, Large White ~6 mo ([PLoS ONE paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3283651/), [J Anat](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1469-7580.2011.01408.x)) | 755 N (masseters 282/178 N, temporalis 195/100 N), supports at teeth and TMJ | 16 rosettes incl. maxilla dorsal to M1 (G3), dorsal maxilla (G4), zygoma (G5) | G3 +388/-375 με; G4 +264/-107 με; G5 +456/-288 με. FE with E=12.5 GPa came out about 10x too stiff. |
| A3 | Human cadaver heads, n=5 ([PubMed 22976651](https://pubmed.ncbi.nlm.nih.gov/22976651/)) | 100 N single muscle | 8 midface gauges | Thin antral bone +685/-722 με; zygoma +209/-78 με |
| A5 | Human cadaver head and 7 FE crania ([PMC4975005](https://pmc.ncbi.nlm.nih.gov/articles/PMC4975005/)) | Simulated bites | Nasal margin and others | In vitro +108/-282 με vs FEA +115/-238 με; r² 0.91-0.95 |
| A6 | Macaque ex vivo ([Kupczik 2007](https://onlinelibrary.wiley.com/doi/10.1111/j.1469-7580.2006.00662.x)) | 15-20 N | 6 gauges | Best E 9.1-14 GPa; worst error ~75 με |
| A8/A9 | Human skull RME ([Boryor 2010](https://pubmed.ncbi.nlm.nih.gov/20170920/), [2013](https://pubmed.ncbi.nlm.nih.gov/23377077/)) | Transverse force to rupture; 86 N at one screw | Zygomatic process strain; screw force | Linear force-strain until rupture; numbers behind paywall |
| A10 | RME screw bench test ([PMC5124561](https://pmc.ncbi.nlm.nih.gov/articles/PMC5124561/)) | Screw activations | Load cell | 4B screw plateau ~160-170 N; 2B plateau ~280 N |

Provatidis 2006 ([DOI](https://doi.org/10.1243/09544119JEIM92)) is the only dry-skull RME validation I found, but I could not see numbers.

## B. In vivo strain during chewing

- **Pig maxilla (B1):** [Herring et al. 2001](https://www.tamug.edu/marshall/Pubs_repository/pig_mastication.pdf), 11 minipigs, gauges on the dorsal non-alveolar maxilla. Tension 206-232 με; compression -18 to -160 με by site. No bite force was reported.
- **Pig diet effect on strain (B2):** [Herring 2005](https://pmc.ncbi.nlm.nih.gov/articles/PMC1810235/), 10 piglets. Zygomatic peak tension was 221 με on hard chow and 122 με on soft chow, about 45% lower.
- **Pig sutures (B3):** [Herring and Mucci 1991](https://pmc.ncbi.nlm.nih.gov/articles/PMC2814820/). Suture strain reached 1,000-2,000 με, higher than bone.
- **Macaque (B4, B5):** [Strait 2005 compilation](https://www.dgp.toronto.edu/~elf/.misc/archeology/Strait%20et%20al.%202005.pdf) gives infraorbital shear 180-613 με. [Ross 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3039785/) gives orientation comparisons.
- **Human (B6):** [Hillam 2015](https://pubmed.ncbi.nlm.nih.gov/26232812/), one subject, skull strains mostly below 100 με. Strain during biting did not depend on bite force.

Note on scale: physiological facial strains are 100-600 με. Frost's thresholds put the disuse window below roughly 100-300 με and modeling above 1,500 με ([Frost 1987](https://onlinelibrary.wiley.com/doi/10.1002/ar.1092190104), snippet only). A model setpoint in the 200-400 με range is therefore consistent with the pig data.

## C. Tongue pressure, swallowing and daily counts

- Adult swallow and maximum pressures: C1. Child swallow and maximum pressures: C2. Resting pressure: C3.
- Swallows per day: about 600 ([Jotz review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10411117/)). This is secondary and the original measurement is unverified. The config cites Lear 1965 (585/day), which I did not retrieve.
- Chewing time averages 35 min/day, range 7-76 ([Sci Adv 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9385136/)). My own arithmetic at 60-80 cycles/min gives roughly 2,100-2,800 cycles/day, above the config value of 1,500 but inside its sweep (200-3,300). **UNVERIFIED**.
- No source gives hours per day of resting tongue-to-palate contact in adolescents. This remains a gap (Needs Wes).

## D. Longitudinal change under altered loading

| ID | Species, n, duration | Outcome and effect |
|---|---|---|
| D1 | Pigs, hard n=9 vs soft n=11, 12 wk ([Burn 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC3384692/)) | Intermaxillary suture growth +13% to +26% with hard diet (p=0.002 pooled); bone quantity score lower with soft diet (p<0.001) |
| D5 | Rats, 8/group, 9 d ([Uysal 2010](https://reference-global.com/download/article/10.2478/aoj-2010-0009.pdf)) | New bone 34.9% (none), 58.8% (10 cycles/day), 70.1% (1,000 cycles/day) |
| D4 | Rabbits, 18, 29 d ([Eur J Orthod](https://academic.oup.com/ejo/article/32/4/371/514183)) | Intermittent force gave 61% of continuous bone formation |
| D2 | Rats, 20, 4 wk ([Fujita 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8612051/)) | Soft diet worsened all maxillary micro-CT measures; numbers not extracted |
| D3 | Rats ([Shimizu 2013](https://www.sciencedirect.com/science/article/abs/pii/S0003996913000617)) | Soft diet caused less osteopenia in maxilla than mandible; numbers not extracted |
| D10 | Humans, 392 adolescents, 1 y ([Macrì 2022](https://ouci.dntb.gov.ua/works/4kd5OBnl/)) | Suture density change by sex and pattern; numbers not extracted |

Caution on D5: this is a cell-and-suture experiment after expansion, not steady-state remodeling. METHODS.md already excludes sutural growth from the strain-adaptive law, so D5 informs the cycle-count dependence only qualitatively.

## E. Open CT/CBCT

- **No adolescent maxilla CBCT with a clearly commercial-friendly license was found.**
- Open-Full-Jaw (E1) is reportedly CC BY-NC-SA 4.0 per a [third-party attribution file](https://github.com/choxos/OMFAtlas/blob/main/public/models/dental/ATTRIBUTION.md); I could not confirm it on the repository itself.
- ToothFairy2 (E2) is CC BY-SA with sign-up, [per its dataset page](https://toothfairy2.grand-challenge.org/dataset/). Patient ages were not checked.
- The FDA pediatric CT set (E3) covers 1 month to 10 years only ([FDA](https://cdrh-rst.fda.gov/normative-dataset-healthy-pediatric-cranial-computed-tomography-ct-images)); terms are not stated.
- A [systematic review of open oral-maxillofacial datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC12228768/) catalogs 105 datasets; screening it for adolescent CBCT is the next step. A [GitHub list](https://github.com/ImaginAR2T/CBCT) of public CBCT sets (CC-BY-4.0 repo) is another lead; I did not open it.

## Recommendations

### 1. Best calibration set
Use **A1/A2 (pig skull, 755 N) plus A3 (human cadaver, 100 N/muscle)** to calibrate cortical and cancellous modulus and boundary conditions. Use **B1 and B2** to set the strain scale of the chewing setpoint (about 200-250 με on the maxilla; zygoma 120-220 με depending on food). Use **C1/C2** for swallow pressure and **C3** for resting pressure. Do not tune against a quantity you later validate against.

### 2. Best held-out validation set
- **Structural:** B1 (pig maxilla in vivo) for the strain level at the three maxillary sites, with the pig FE built from the same animal species. Pass criterion suggestion: the model falls within the reported ±SD at each site. This is my proposal, not from a source.
- **Adaptive:** D1 (pig hard vs soft diet, 12 wk) for the sign and rough size of the density response. The model should predict lower density under a soft-diet loading history (about 45% lower zygomatic strain per B2), with a magnitude in the same direction as the +13-26% suture growth difference and bone-quantity decrease in D1 (different endpoints, so only sign and rough order are testable). D5 adds the cycle-count dependence.
- **Human:** D10 only after the author data request, since only aggregate density is in the paper.

### 3. Pig-maxilla ex vivo bench test to replicate
Replicate **A1/A2**: fresh defleshed pig skull, 247×141×133 mm, six months, with 16 rosette gauges. Apply 755 N through the four muscle sites (282/178 N masseters, 195/100 N temporalis), support at teeth and TMJs, three repeated loadings. Compare model and gauges at G3, G4 and G5. A cheap alternative is to scale to about 300 N for specimen safety (UNVERIFIED that this is within elastic range). Add a DIC speckle pattern on the maxilla for full-field comparison, which A1/A2 lacked. G1 (premaxilla) and G6 (posterior nasal) were unstable in the original, so place extra gauges there. Skull supply: pig heads from a butcher are cheap, but donor age should be recorded because Herring found age effects.

## Needs Wes

1. **Send the author data requests** in `AUTHOR_REQUEST_DRAFTS.md` (not sent).
2. **Decide the context of use.** Pig ex vivo and in vivo data are non-human adults/juveniles, and the adolescent human maxilla has no direct strain data. State this limitation in the preprint.
3. **Confirm licenses by reading the repositories** for Open-Full-Jaw and ToothFairy2. I relied on third-party and snippet text.
4. **Bench test needs hardware and an institution:** strain gauges, a load frame and a specimen source. Cost and ethics/biosafety approval are your call.
5. **Paywalled papers to read through a library:** Boryor 2010, Provatidis 2006, Macrì 2022, Shimizu 2013, Davami 2020, Hylander and Johnson 1997.
6. **Mewing scenario (2 kPa) has no measurement behind it.** Keep it labeled a hypothesis, or find a palatal sensor study with deliberate posture.
7. **Citation check for the other session:** `configs/study.yaml` cites Lear 1965; I did not retrieve it, so that citation is unchecked.
