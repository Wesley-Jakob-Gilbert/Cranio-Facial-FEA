# Prior Work Review: Craniofacial FE Validation and Tongue-Posture Mechanobiology

Status: literature review for Paper 1 (bench validation of CT-based FE strain predictions in the pig maxilla/palate under 10-70 kPa hydrostatic palatal loading, frozen predictions, three solvers, open data) and Paper 2 (can resting tongue posture, ~0.1-2 kPa, drive maxillary remodeling compared with chewing/swallowing).
Companion table: `docs/validation/prior_work.csv` (85 rows; one row per source; `source_read` column = FT full text read, ABS abstract only, SECONDARY value taken from a citing review).
Written under TEAM_BRIEF rules: nothing outside `docs/validation/` and `validation/` was touched; nothing committed.

## How to read this document

- **Verification vs validation.** Verification means checking the equations are solved correctly (code-to-code, mesh convergence, analytical benchmarks). Validation means checking the model against physical measurements. This follows [Anderson, Ellis & Weiss 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC3361760/). Paper 1's three-solver comparison is **verification**. The strain-gauge comparison is **validation**.
- **UNVERIFIED** means only the abstract (or a secondary source) was read. Numbers marked that way should be checked against full text before they go into a manuscript.
- **Method.** Searches covered an academic index (Semantic Scholar-style), Europe PMC REST (metadata and abstracts for ~100 DOIs; scratch in `validation/litreview/meta.json`), PubMed/PMC pages, publisher pages and general web search. Open-access full texts were read for the sources marked FT. Scratch outputs are in `validation/litreview/`.
- **Scope of "not found".** "No study found" means none turned up in these searches. It does not prove absence. A systematic search with PRISMA-style logging would be needed before claiming "first" in a manuscript.

---

## A. Craniofacial FE validation studies

### A1. Ex vivo / in vitro strain validation (most relevant to Paper 1)

| Study | Species, n | Loading | Measurement | Materials / solver | Agreement result | Lesson |
|---|---|---|---|---|---|---|
| [Bright & Rayfield 2011, J Anat](https://doi.org/10.1111/j.1469-7580.2011.01408.x) (FT) | Domestic pig (~6 mo), **n=1** | Muscle-direction loads, 755 N total | 16 rosettes (2 excluded) | Homogeneous 12.5 GPa, plus cortical-shell/cancellous variants; Abaqus 6.8.2, ~1.75M TET10 | Homogeneous model was about an order of magnitude too stiff. Most orientations were within 10 deg; 4 gauges were off by 25-40 deg. The cortical/cancellous model improved magnitudes | Closest precedent. A homogeneous 12.5 GPa model (our M0) is expected to fail on magnitude |
| [Bright 2012, PLoS ONE](https://doi.org/10.1371/journal.pone.0031769) (FT) | Same pig | Same | Same | Sutures at 46 MPa | Sutures did **not** improve global fit (strain-ratio distance 3.78 homogeneous vs 4.86 with sutures). Experimental SE 5-10% | Report suture sensitivity rather than assume it helps |
| [Bright & Gröning 2011, J Morphol](https://doi.org/10.1002/jmor.10991) (ABS) | Pig zygomatic arch | Ex vivo | DSPI full-field | Suture variants | A compliant solid suture matched best (UNVERIFIED) | Full-field optics have been done on pig |
| [Kupczik et al. 2007, J Anat](https://doi.org/10.1111/j.1469-7580.2006.00662.x) (ABS) | *M. fascicularis*, n=2 | Non-physiological arch loads | Gauges | Nanoindentation properties; voxel FE (UNVERIFIED) | Measured properties gave high congruence. Including the suture improved fit. Muscle-applied vs node-applied loads gave distinctly different strains | How the load is applied matters. Hydrostatic loading removes most of this uncertainty |
| [Szwedowski et al. 2010, J Craniofac Surg](https://doi.org/10.1097/SCS.0b013e3181c50f64) (ABS) | Human cadaver, n=1 | Masseter-direction <150 N, 4 orientations | 12 sites / 16 channels | Shell model, I-deas | r=0.57 overall; 0.70 excluding assumption-heavy regions (UNVERIFIED) | Use pre-test FE to place gauges |
| [Szwedowski et al. 2011, Proc IMechE H](https://doi.org/10.1243/09544119JEIM786) (ABS) | Same | Same | Same | Local cortical thickness | r^2=0.73 with local thickness (UNVERIFIED) | Thin-bone thickness is critical |
| [Ledogar et al. 2016, PeerJ](https://doi.org/10.7717/peerj.2242) (FT) | Human cadaver (42 y), n=1, 3 trials | In vitro P3 bite | 14 sites | Cortical E 17.92-25.52 GPa; Strand7 | Log regression slope 0.845, R^2=0.909 (max principal); slope 0.849, R^2=0.953 (min principal). Site errors 0.79-80%, with the large % errors at low-strain sites | Realistic target. Percent error blows up at low strain, so use log-ratio or absolute error |
| [Toro-Ibacache et al. 2016, J Anat](https://doi.org/10.1111/joa.12384) (ABS) | Human cranium, n=1 | Incisor bite | DSPI | Cancellous-as-cortical; thin-bone omission | Filling cancellous with cortical lowered strains. Omitting thin circum-nasal bone changed both mode and magnitude (UNVERIFIED) | Thin bone and cancellous treatment drive magnitude |
| [Gröning et al. 2009, J Biomech](https://doi.org/10.1016/j.jbiomech.2009.03.025) (ABS) | Human mandible | Ex vivo | DSPI | Voxel FE | Good agreement reported (UNVERIFIED) | Full-field validation is feasible |
| [Gröning et al. 2012, J Biomech](https://doi.org/10.1016/j.jbiomech.2012.02.009) (ABS) | Human mandible | Ex vivo | Quantitative full-field DSPI | Simplified | Agreement varied within and between regions (UNVERIFIED) | Report results per region |
| [Marinescu et al. 2005, Anat Rec](https://doi.org/10.1002/ar.a.20166) (ABS) | Macaque mandible | In vitro | Gauges | Isotropic vs heterogeneous orthotropic | Isotropic fit poorly. Small load-direction errors had large effects | Boundary conditions and loads dominate |
| [Cuff et al. 2015, PeerJ](https://doi.org/10.7717/peerj.1294) (FT) | Ostrich, n=1 | ~25 N per side | 12 gauges | Abaqus Tet10 ~1.75M | Magnitudes not reliably reproduced. Convergence criteria were <10% strain and <5% SED | Thin-walled skulls are hard. Precedent for convergence targets |
| [Rayfield 2011, J Anat](https://doi.org/10.1111/j.1469-7580.2010.01296.x) (ABS) | Ostrich mandible, n=3 | Simulated pecking | 4 gauges | Specimen-specific | Mixed (UNVERIFIED) | Rare multi-specimen precedent |
| [Casas et al. 2007, Dent Traumatol](https://doi.org/10.1111/j.1600-9657.2006.00460.x) (ABS) | Human cadaver maxilla | Impact | Rosettes **including medial palate** | ~74,000 DOF | 1.7-11.4% difference (UNVERIFIED) | The only FE comparison with a palatal gauge we found. Load type differs |

### A2. In vivo strain vs FE

| Study | Key result |
|---|---|
| [Ross et al. 2011, J Anat](https://pmc.ncbi.nlm.nih.gov/articles/PMC3039785/) (FT): 4 *M. mulatta* + 1 *Chlorocebus* in vivo; Strand7, 311,047 elements | Raw FE strains fell below in vivo ranges at 6/20 sites. After normalisation most were within range. Some orientations mismatched |
| [Panagiotopoulou et al. 2017, Zoology](https://pmc.ncbi.nlm.nih.gov/articles/PMC5792078/) (FT): 1 rhesus, 163 chews, 3 rosettes, Abaqus 6.13, 80 orthotropic regions | In vivo e1/e2 vs model: 81/-90 vs 127/-98 (ULAT), 56/-48 vs 80/-70 (LLAT), 50/-69 vs 32/-41 (MED). TMJ and bite constraints had the strongest effect |
| [Mehari Abraha et al. 2019, Front Bioeng Biotechnol](https://doi.org/10.3389/fbioe.2019.00269) (FT) | PDL modulus over 0.07-1750 MPa had little global effect. 2/3 sites fell within 1.5xIQR |
| [Strait et al. 2005, Anat Rec](https://doi.org/10.1002/ar.a.20172) (ABS) | Regional orthotropic properties deformed most realistically |
| [Porro et al. 2013, J Anat](https://doi.org/10.1111/joa.12080) (ABS) | Alligator: orientations matched closely, but FE magnitudes were lower |
| Note | A 2011 Panagiotopoulou PDL paper was retracted ([retraction notice](https://doi.org/10.1111/joa.12322)). Do not cite it |

### A3. Maxillary-expansion and pig maxilla measurement studies

- [Provatidis et al. 2006](https://doi.org/10.1243/09544119JEIM92) (ABS): dry human skull with 4-point displacement vs FE. Suture ossification dominated the displacement distribution.
- [Boryor et al. 2010](https://doi.org/10.1016/j.jbiomech.2010.02.002) (ABS): fresh and Thiel-fixed skulls showed linear behaviour until rupture. This supports linear elasticity at low loads.
- [Rivera-Tapia et al. 2025 review](https://doi.org/10.1016/j.jbiomech.2025.113010) (ABS): only 6 of 79 MARPE FE studies included clinical validation.
- Pig ex vivo maxilla instrumentation with no FE: [Sun et al. 2011](https://doi.org/10.1016/j.ajodo.2009.12.029) (8 heads, strain linear with activation, 20-30% decay, ABS) and [Fox et al. 2020](https://doi.org/10.1111/ocr.12384) (28 heads, DVRT across the midpalatal suture plus gauges, ABS). Together they show gauges can be bonded on the pig maxilla, and that viscoelastic decay occurs.
- [Valeri et al. 2024, Heliyon](https://doi.org/10.1016/j.heliyon.2024.e24952) (FT): butcher-sourced pig palates for miniscrew tests, ANSYS model, no strain validation.

### A4. Sensitivity and reviews

- [Walmsley et al. 2013, PeerJ](https://doi.org/10.7717/peerj.204) (FT): heterogeneous vs homogeneous material changed results by 1.71-11.86%. Bite position changed them by 27-66%. Loading definition outweighs material choice.
- [Wang et al. 2010, Anat Rec](https://doi.org/10.1002/ar.21203) (ABS): suture stiffness mainly changes strain locally, near the suture.
- [Soodmand et al. 2024, Heliyon](https://doi.org/10.1016/j.heliyon.2024.e40668) (FT): 77 mandible FE studies used 12 different density-modulus relations, with no consensus.
- [Peterson et al. 2006, Anat Rec](https://doi.org/10.1002/ar.a.20358) (ABS): across 15 human maxillae, properties were more variable than in the mandible.

### A5. Gaps in A
- No CT-FE strain validation was found for rat, dog or sheep craniofacial bone.
- No craniofacial validation was found that uses distributed palatal pressure.
- Almost all craniofacial validations use n=1 specimen.

---

## B. CT-based bone FE validation methods (non-craniofacial benchmarks)

| Study | Design | Result | Lesson for Paper 1 |
|---|---|---|---|
| [Keyak et al. 1993](https://doi.org/10.1016/0141-5425(93)90066-8) (ABS) | 1 femur, 11 gauges | Early automated CT-FE validation | Historical baseline |
| [Schileo et al. 2007](https://doi.org/10.1016/j.jbiomech.2007.02.010) (ABS) | 8 femurs, 15 rosettes, 6 loads, 3 density-elasticity laws | Best law: R^2=0.91, RMSE <10% of max | The density-elasticity law dominates accuracy |
| [Schileo et al. 2008](https://doi.org/10.1016/j.jbiomech.2008.05.017) (ABS) | Ash-density calibration | R^2=0.95, RMSE 7% | Use a phantom-calibrated CT and convert to ash density |
| [Taddei et al. 2007](https://doi.org/10.1016/j.medengphy.2006.10.014) (ABS) | 1 femur, 13 gauges, 5 loads | Averaging modulus per voxel instead of density raised strain R from 0.69 to 0.79 and cut mean error from 45.1% to 31.3% | Average modulus, not density, in M2 |
| [Schileo et al. 2020, Bone](https://doi.org/10.1016/j.bone.2020.115348) (ABS) | 14 femurs | Cortical bone mapping cut error by ~30% | Thin cortex needs special handling; the palate is thin cortex |
| [Yosibash et al. 2007](https://doi.org/10.1115/1.2720906); [Dahan et al. 2016](https://doi.org/10.1016/j.jbiomech.2016.02.036) (ABS) | p-FE femur and humerus | Humeri: slope 1.09, R^2=0.98; p-FE and h-FE similar | Converged discretisations agree, so element order matters only before convergence |
| [Grassi et al. 2016](https://doi.org/10.1016/j.jbiomech.2016.02.032) (ABS) | 3 femora, DIC | R^2=0.93, RMSE 10% | DIC is a viable full-field validation |
| [Kok et al. 2021](https://doi.org/10.1016/j.jbiomech.2021.110445) (ABS) | 12 femora | Strain R^2 0.87 / 0.74 | Plausible agreement range |
| [Fung et al. 2017](https://doi.org/10.1016/j.jbiomech.2017.06.010) (ABS) | 33 metatarsals | r^2 >=0.94 | Large-n precedent |
| [Garavelli et al. 2022, PLoS ONE](https://doi.org/10.1371/journal.pone.0272529) (FT) | Lumbar spine segment, DIC, ANSYS, E=4730 rho^1.56 | Displacement R^2 0.94-0.98; slopes 0.65-1.07 | Slopes spread even in good models, which supports the 0.80-1.25 acceptance window |

**Frameworks.**
- [Anderson et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC3361760/) (FT): define intended use first, verify before validating, and run mesh convergence on the validation outputs.
- [Henninger et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2941217/) (FT): correlation is not agreement. Use Bland-Altman and RMSE, and list error sources.
- [Erdemir et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3278509/) (FT): reporting checklist.
- [ASME V&V 40-2018](https://www.asme.org/codes-standards/find-codes-standards/assessing-credibility-of-computational-modeling-through-verification-and-validation-application-to-medical-devices): a risk-informed credibility framework. It explicitly does not prescribe quantitative thresholds or specific activities.
- [FDA 2023 CM&S guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/assessing-credibility-computational-modeling-and-simulation-medical-device-submissions): risk-informed, applies to physics-based models.
- [Nagaraja et al. 2024](https://doi.org/10.1016/j.ymeth.2024.03.003): an end-to-end V&V 40 example.

**Blinded prediction.** [Dunbar et al. 2025](https://doi.org/10.1016/j.medengphy.2025.104321) (ABS) used blinded FE prediction of a clinical implant failure. It was qualitative and in the pelvis. We found no pre-registered or blinded quantitative strain validation in craniofacial FE.

---

## C. Multi-solver comparisons and FEBio verification

| Study | Solvers | Result | Implication |
|---|---|---|---|
| [MacLeod, Rose & Gill 2016, J Biomech Eng](https://doi.org/10.1115/1.4034653) (ABS, [PubMed](https://pubmed.ncbi.nlm.nih.gov/27618586/)) | Abaqus, ANSYS, FEBio; composite femur, 4 gauges | Solvers agreed within 3.0% on average (r^2 >0.99); the two commercial codes within 1.4%. RMSE vs experiment was 134.5% for a generic model and 13.8% for specimen-specific models. Model and data released openly | **Direct precedent** for "multi-solver + open data". Solver choice mattered far less than geometry |
| [Maas et al. 2012, FEBio](https://pmc.ncbi.nlm.nih.gov/articles/PMC3705975/) (FT) | FEBio vs NIKE3D/ABAQUS/analytical | Verification suite (agreement level per our extraction; check exact figure) | FEBio verification baseline exists |
| [Meng et al. 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3834732/) (FT) | Abaqus 6.9-EF1 vs FEBio 1.5.0, biphasic contact | Mean differences 0.3-7.6% | Differences appear with contact and nonlinearity |
| [Paz et al. 2023, Ann Biomed Eng](https://doi.org/10.1007/s10439-023-03252-8) (FT) | Abaqus 2018 vs FEBio 3.3.2, 154 knees | Abaqus peak stresses were 27% higher and overstressed volume ~43% greater; classification AUC was the same (0.73) | Large differences occur when constitutive models are not identical |
| [Erdemir et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6611350/), [Rooks et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8086182/) (FT) | 5 teams, mixed codes | Phases were pre-specified and deviations documented; modeller choices drove the spread | Precedent for pre-specification |
| [Halloran et al. 2023, J Orthop Res](https://doi.org/10.1002/jor.25358) (ABS) | Literature audit | Reporting, reproducibility and verification scores were low (UNVERIFIED values in CSV) | Thorough verification reporting is an easy differentiator |
| [Rooks et al. 2022](https://doi.org/10.3389/fbioe.2022.841882) (ABS) | FEBio 3.0 | Contact settings had large effects | Prefer bonded/tied fixtures in the bench model |
| [Schneider et al. 2022, ACM TOG](https://doi.org/10.1145/3508372) ([arXiv](https://arxiv.org/abs/1903.09332)) | PolyFEM, tet vs hex | Large-scale benchmark of FEM pipelines | Supports an explicit tet4/tet10 study |
| [Open-Full-Jaw 2022](https://doi.org/10.1016/j.cmpb.2022.107009) (ABS) | 17 human jaw models | Geometry and mesh only; no experimental validation | Open craniofacial **models** exist, but open craniofacial **strain data** do not (search-limited). Licence reportedly CC BY-NC-SA 4.0 per TEAM_BRIEF |

**C gap.** We found no multi-solver craniofacial comparison. For small-strain linear elasticity on an identical mesh, solvers are expected to agree within a few percent ([MacLeod 2016](https://doi.org/10.1115/1.4034653)). So a three-solver result mainly verifies our pipeline (especially the custom JAX solver). It does not validate the model.

---

## D. Palatal and tongue pressure, myofunctional therapy, "mewing"

### D1. Pig in vivo tongue loading (Herring/Liu group)
- [Liu, Shcherbatyy & Perkins 2008, J Oral Maxillofac Surg](https://pmc.ncbi.nlm.nih.gov/articles/PMC2497442/) (FT): 12 Yucatan minipigs. Palatal and lingual pressures during chewing were 2.12-8.04 kPa. Bone strains were mostly <100 microstrain, with ~200 at the premaxillary suture. Tongue reduction lowered strain by 18-48%.
- [Deng et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC7508637/) (FT): 22 pigs, hypoglossal stimulation. Pressures were ~1-10 kPa and strains <100 microstrain.
- [Liu et al. 2008, Arch Oral Biol](https://pmc.ncbi.nlm.nih.gov/articles/PMC2562725/) (FT): a ~15% tongue reduction slowed anterior growth.
- [Liu et al. 2026, J Morphol](https://doi.org/10.1002/jmor.70144) (FT): in adult minipigs, tongue-base reduction (n=6) narrowed maxillary intercanine width (p<0.001).
- [Rafferty et al. 2003](https://doi.org/10.1002/jmor.10104) (ABS) and [Rafferty et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7232983/) (FT): suture strains are an order of magnitude larger than strains in adjacent bone.

### D2. Pressure-dose evidence on palatal bone (the most important finding for Paper 2)
- [Sato et al. 1998, J Dent Res](https://doi.org/10.1177/00220345980770020701) (ABS; 85 rats, denture-applied palatal pressure): resorption occurred in all rats at >=6.86 kPa continuous or >=19.6 kPa intermittent pressure. The no-resorption limits (<1.96 kPa continuous, <9.8 kPa intermittent) are garbled in the abstract. They are taken from the [Chen et al. 2015 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC4535403/) (SECONDARY).
- [Imai et al. 2002, J Oral Rehabil](https://doi.org/10.1046/j.1365-2842.2002.00799.x) (ABS; 250 rats): continuous 1.0 kPa behaved like 0 kPa, with no resorption and bone formation inhibited up to 4 weeks by denture coverage. 10 and 20 kPa caused 24+/-17 and 35+/-21 um of resorption.
- Implication: sustained palatal pressure in the resting-tongue range (~0.1-2 kPa) sits below the measured resorption thresholds in rat. Its action is through mucosa onto the bone surface, which is not the same pathway as bending strain.

### D3. Equilibrium and dentoalveolar effects
- [Proffit 1978, Angle Orthod](https://doi.org/10.1043/0003-3219(1978)048<0175:ETRFIP>2.0.CO;2) (ABS): long-duration resting pressures, not brief functional forces, determine tooth position.
- [Boosalis Toaddy et al. 2026, PLoS ONE](https://doi.org/10.1371/journal.pone.0353500) (ABS): in mandibular FE, tongue pressure alone expanded the arch, while combined resting loads produced crowding.

### D4. Prior FE of tongue pressure (weak)
- [Melnyk et al. 2026a, Med Sci Ukraine](https://doi.org/10.32345/2664-4738.1.2026.14) (ABS): static tongue pressure, normal vs +25% tongue volume. Stress at normal volume was described as insignificant. The abstract gives no pressure magnitudes, material values or remodeling law.
- [Melnyk et al. 2026b, Kharkiv Dental J](https://doi.org/10.26565/3083-5607-2026-7-04) (ABS): qualitative stress-concentration sites only.
- [Lazarev et al. 2026, Wiad Lek](https://pubmed.ncbi.nlm.nih.gov/42536943/) (ABS): FlexiForce measurements on jaw prototypes; no force values in the abstract.
- [Alsaleem et al. 2025](https://doi.org/10.4103/jispcd.jispcd_38_25) (FT): cortical E of 1.3 MPa, which is implausible. Example of what to avoid.

### D5. Clinical and posture evidence
- [McFarland et al. 2026](https://doi.org/10.1177/15257401261425962) (ABS): 7 studies. Low or anterior resting tongue posture was associated with open bite and crossbite. This is association only.
- [Stefani et al. 2025, Eur J Orthod](https://doi.org/10.1093/ejo/cjaf024) (ABS): 24 studies including 4 RCTs; no Level 1 evidence that myofunctional therapy changes outcomes.
- [Xiao & Ma 2026, Front Pediatr](https://doi.org/10.3389/fped.2026.1809806) (ABS): pilot study, n=64, intercanine width +1.55 mm vs control, quasi-random allocation.
- **Mewing:** no controlled peer-reviewed study was found. Lay summaries say the same, e.g. [Health.com](https://www.health.com/mewing-7098082). This is a negative search result, not proof of no effect.

### D6. Measurement methods
- [Jeannin & Perrier 2023](https://doi.org/10.1016/j.medengphy.2023.104036) (ABS): a water-filled latex finger at 0-2.5 kPa shows a Mullins effect. Precondition the latex dam before testing.
- [Hillam et al. 2015](https://doi.org/10.1016/j.jbiomech.2015.06.021) (FT): in one human subject, peak parietal strain was <=192 microstrain across activities.

---

## E. Strain-adaptive remodeling in craniofacial bone

- [Lanyon & Rubin 1984, J Biomech](https://doi.org/10.1016/0021-9290(84)90003-4) (ABS): a static 525 N load produced bone loss similar to disuse (-13% cross-sectional area). A single 100 s/day bout of intermittent 1 Hz loading at the same load was the comparison condition; it is reported as osteogenic, but that part of the abstract is truncated (UNVERIFIED). **Key for Paper 2:** static load does not substitute for dynamic stimulus.
- [Reina et al. 2007](https://doi.org/10.1016/j.jbiomech.2006.03.007) (ABS): an anisotropic remodeling law applied to the mandible gave reasonable agreement with clinical density.
- [Chou et al. 2015, CMBBE](https://doi.org/10.1080/10255842.2013.792916) (ABS): a stress- and cycle-dependent law with mastication, speech and open-close cycles reproduced tubular architecture. Edentulism led to up to 1/3 mandibular mass loss. This is the closest template for a posture-vs-chewing daily stimulus.
- [Kupczik et al. 2009, AJPA](https://doi.org/10.1002/ajpa.20972) (ABS): SED was low in the supraorbital torus, where bone grows, so local strain does not explain all morphology.
- [Lieberman et al. 2004, J Hum Evol](https://doi.org/10.1016/j.jhevol.2004.03.005) (ABS): raw diet produced up to 2x the strain, and cooked-diet hyraxes had ~10% less facial growth in high-strain regions.
- Growth models from the Moazen group: [Liang et al. 2024, iScience](https://doi.org/10.1016/j.isci.2024.110617), [Liang et al. 2024, Proc R Soc B](https://doi.org/10.1098/rspb.2024.0654) and [Marghoub et al. 2018, J Anat](https://doi.org/10.1111/joa.12764). Details UNVERIFIED.
- Soft-tissue-mediated remodeling:
  - [Sarrafpour et al. 2013](https://doi.org/10.1371/journal.pone.0058803) (ABS): the dental follicle senses bite-induced strain.
  - [Chen et al. 2014](https://doi.org/10.1016/j.jbiomech.2014.02.030) (ABS): a PDL hydrostatic-stress surface-remodeling law matched clinical tooth movement.
  - Together these support a different mechanism for sustained light loads.
- **Gap:** no study was found that applies a strain-adaptive remodeling law to resting tongue posture vs chewing in the maxilla.

---

## Gap analysis

### (1) Paper 1: has it been done?

**Verdict: partly done. Novelty is moderate, and thin if framed as "first validation of pig craniofacial FE".**

**Already done:**
- Pig cranium plus CT-FE plus rosettes ([Bright & Rayfield 2011](https://doi.org/10.1111/j.1469-7580.2011.01408.x); n=1, muscle loads).
- Pig full-field optics ([Bright & Gröning 2011](https://doi.org/10.1002/jmor.10991)).
- Three-solver agreement plus open data on bone ([MacLeod 2016](https://doi.org/10.1115/1.4034653)).
- Material-mapping comparisons ([Schileo 2007](https://doi.org/10.1016/j.jbiomech.2007.02.010), [Taddei 2007](https://doi.org/10.1016/j.medengphy.2006.10.014)).

**Apparently new (search-limited):**
- (a) Distributed hydrostatic palatal loading with a known pressure field. The only palatal-gauge comparison found is an impact study ([Casas 2007](https://doi.org/10.1111/j.1600-9657.2006.00460.x)).
- (b) n=4 specimens. Most craniofacial validations are n=1.
- (c) Predictions frozen before testing. None found in craniofacial work. Pre-specification exists in knee modeling ([Erdemir 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6611350/)).
- (d) An open craniofacial strain dataset.
- (e) A low-cost rig.

**Weak points:**
- The three-solver comparison is verification. Agreement is expected for linear elasticity ([MacLeod 2016](https://doi.org/10.1115/1.4034653), [Meng 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3834732/)), so it is not a headline result.
- Excised pig palates at 10-70 kPa are not a physiological load case. The pressures are far above the 2-8 kPa chewing pressures measured in pigs ([Liu 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2497442/)).

**How to sharpen:**
1. Frame it as a **controlled-load isolation experiment**. Prior errors are dominated by load and boundary-condition uncertainty ([Marinescu 2005](https://doi.org/10.1002/ar.a.20166), [Walmsley 2013](https://doi.org/10.7717/peerj.204), [Panagiotopoulou 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5792078/)). Hydrostatic loading removes most of the load uncertainty, so the remaining error can be attributed to material and geometry (M0/M1/M2).
2. Submit as a **Registered Report**. [RSOS](https://royalsocietypublishing.org/rsos/registered-reports) gives Stage 1 in-principle acceptance and allows normally one year to submit results. This turns "frozen predictions" into a verifiable claim.
3. Make it a **benchmark dataset** with CT, meshes, gauge positions and raw data, as in [MacLeod 2016](https://doi.org/10.1115/1.4034653).
4. Add an element-order study (tet4 vs tet10) and a boundary-condition sensitivity study.
5. Add DIC on the oral surface, if feasible, following [Grassi 2016](https://doi.org/10.1016/j.jbiomech.2016.02.032) and [Gröning 2012](https://doi.org/10.1016/j.jbiomech.2012.02.009).

### (2) Paper 2: has it been done?

**Verdict: not done as specified, but the headline conclusion is largely predictable. Novelty is real but thin unless reframed.**

**What exists:**
- Static tongue-pressure FE with no remodeling law and no stated magnitudes ([Melnyk 2026a](https://doi.org/10.32345/2664-4738.1.2026.14), [2026b](https://doi.org/10.26565/3083-5607-2026-7-04)).
- Mandibular dentition FE ([Boosalis Toaddy 2026](https://doi.org/10.1371/journal.pone.0353500)).
- Jaw remodeling with daily stimuli ([Chou 2015](https://doi.org/10.1080/10255842.2013.792916)).

None compares resting posture with chewing in the maxilla using a strain-adaptive law.

**The problem.**
- Standard strain-adaptive laws weight cyclic and dynamic stimulus. Static loads behave like disuse ([Lanyon & Rubin 1984](https://doi.org/10.1016/0021-9290(84)90003-4)).
- Tongue loading even during function yields <100 microstrain in pigs ([Liu 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2497442/), [Deng 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC7508637/)).
- So "posture contributes negligibly" is close to an assumption of the model rather than a finding.

**How to sharpen:**
1. Do a **"what would have to be true"** inversion. Find the static-stimulus weighting, threshold, or pressure-duration product at which posture would matter, and compare it with measured thresholds.
2. Model the **alternative mechanisms** explicitly:
   - Pressure-mediated surface resorption and apposition, with thresholds from rat palate data ([Sato 1998](https://doi.org/10.1177/00220345980770020701), [Imai 2002](https://doi.org/10.1046/j.1365-2842.2002.00799.x)).
   - Sutural strain, since sutures strain an order of magnitude more than bone ([Rafferty 2003](https://doi.org/10.1002/jmor.10104)).
   - Dentoalveolar equilibrium ([Proffit 1978](https://doi.org/10.1043/0003-3219(1978)048<0175:ETRFIP>2.0.CO;2)).
3. **Test the model's sign** against tongue-reduction data in pigs ([Liu 2008 AOB](https://pmc.ncbi.nlm.nih.gov/articles/PMC2562725/), [Liu 2026](https://doi.org/10.1002/jmor.70144)) and soft-diet data ([Lieberman 2004](https://doi.org/10.1016/j.jhevol.2004.03.005)).
4. State explicitly that **no controlled "mewing" evidence** was found. Frame the paper as mechanistic plausibility, not a clinical claim.

### (3) Protocol design lessons (for BENCH_TEST_PROTOCOL; suggestions only, not edits)

1. **Expect magnitude errors. Make material and thickness mapping the primary hypothesis.**
   - M0 (homogeneous 12.5 GPa) failed by about 10x on pig ([Bright & Rayfield 2011](https://doi.org/10.1111/j.1469-7580.2011.01408.x)).
   - Thin bone and cancellous treatment drive magnitudes ([Toro-Ibacache 2016](https://doi.org/10.1111/joa.12384), [Szwedowski 2011](https://doi.org/10.1243/09544119JEIM786)).
   - For M2: use phantom calibration to ash density ([Schileo 2008](https://doi.org/10.1016/j.jbiomech.2008.05.017)), voxel-modulus averaging ([Taddei 2007](https://doi.org/10.1016/j.medengphy.2006.10.014)) and cortical mapping for the thin palate ([Schileo 2020](https://doi.org/10.1016/j.bone.2020.115348)).
   - Measure palatal cortical thickness at each gauge site, from micro-CT or sectioning after the test.
2. **Use quadratic tetrahedra (TET10) and check convergence on the gauge-site outputs.**
   - Prior pig and ostrich validations used TET10 ([Bright & Rayfield 2011](https://doi.org/10.1111/j.1469-7580.2011.01408.x), [Cuff 2015](https://doi.org/10.7717/peerj.1294)).
   - Convergence target: <5% change at gauge sites ([Anderson 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC3361760/)).
   - Treat tet4 as a reported sensitivity case only.
3. **Control and measure boundary conditions, and choose low-gradient gauge sites.**
   - Constraints dominate error ([Panagiotopoulou 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5792078/), [Marinescu 2005](https://doi.org/10.1002/ar.a.20166)).
   - Instrument the pot (dial gauge or DIC) and run a BC sensitivity analysis (fully fixed vs compliant Bondo).
   - Use the frozen FE to place gauges away from strain gradients and sutures ([Szwedowski 2010](https://doi.org/10.1097/SCS.0b013e3181c50f64); [Bright & Rayfield 2011](https://doi.org/10.1111/j.1469-7580.2011.01408.x) site G5; [Rafferty 2003](https://doi.org/10.1002/jmor.10104)).

Additional lessons:
- Add Bland-Altman and log-ratio agreement measures. R^2 and slope alone are not enough ([Henninger 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2941217/)).
- Low-strain channels inflate percent error ([Ledogar 2016](https://doi.org/10.7717/peerj.2242)), so apply the ±30% criterion only above a strain floor.
- Report orientation and magnitude separately. Orientation usually agrees better ([Porro 2013](https://doi.org/10.1111/joa.12080), [Bright & Rayfield 2011](https://doi.org/10.1111/j.1469-7580.2011.01408.x)).
- Precondition the latex dam ([Jeannin & Perrier 2023](https://doi.org/10.1016/j.medengphy.2023.104036)) and allow for viscoelastic decay ([Sun 2011](https://doi.org/10.1016/j.ajodo.2009.12.029)).
- Realistic targets from the literature: R^2 ~0.73-0.95 and slopes ~0.85-1.1 ([Ledogar 2016](https://doi.org/10.7717/peerj.2242), [Schileo 2007](https://doi.org/10.1016/j.jbiomech.2007.02.010), [Garavelli 2022](https://doi.org/10.1371/journal.pone.0272529)). The current acceptance criteria are ambitious but not unrealistic for M2. They are likely unachievable for M0.

### (4) Venues and people

**Venues.** Fit judgments are based on where the closest prior work was published; scope claims are otherwise UNVERIFIED.
- Paper 1: J Anat (Bright, Kupczik, Ross, Toro-Ibacache), J Biomech (Schileo, Gröning), J Biomech Eng (MacLeod 2016, FEBio), PeerJ (Ledogar, Cuff), PLoS ONE, or the [RSOS Registered Report](https://royalsocietypublishing.org/rsos/registered-reports) route.
- Paper 2: Biomech Model Mechanobiol or CMBBE (Chou 2015 published in CMBBE), J Anat, or J R Soc Interface (the Chen 2015 mucosa review appeared there).
- Preprints on bioRxiv or engrXiv. Data and frozen predictions on OSF or Zenodo, with a timestamped pre-registration.

**Researchers to contact** (relevance from the papers above; no contact details verified):
1. **Zi-Jun Liu / Susan Herring** (Univ. Washington): pig tongue pressure, palatal strain and tongue-reduction growth ([Liu 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2497442/), [Liu 2026](https://doi.org/10.1002/jmor.70144)). Key for both papers; possibly in vivo data.
2. **Jen Bright / Emily Rayfield** (Hull / Bristol): pig cranial ex vivo validation ([Bright & Rayfield 2011](https://doi.org/10.1111/j.1469-7580.2011.01408.x)). They may share specimen and gauge data.
3. **Enrico Schileo / Fulvia Taddei** (Rizzoli): density mapping and validation metrics ([Schileo 2008](https://doi.org/10.1016/j.jbiomech.2008.05.017), [Taddei 2007](https://doi.org/10.1016/j.medengphy.2006.10.014)).
4. **Steve Maas / Jeff Weiss** (Utah, FEBio): verification and solver comparison ([Maas 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3705975/)).
5. **Mehran Moazen** (UCL): craniofacial growth and remodeling models ([Liang 2024](https://doi.org/10.1016/j.isci.2024.110617)). Key for Paper 2.

Alternates: Callum Ross / Olga Panagiotopoulou (in vivo strain plus FE), and Qing Li / Michael Swain (Sydney; oral mucosa and jaw remodeling).

---

## Corrections to existing docs (flag only; not edited)

- `DATA_SOURCES.md` row A5 cites PMC4975005. That PMC ID is [Ledogar et al. 2016, PeerJ "Human feeding biomechanics"](https://doi.org/10.7717/peerj.2242): human cadaver, 14 sites, slope 0.845 / R^2 0.909. It is **not** Toro-Ibacache 2016, which is [J Anat 228:70](https://doi.org/10.1111/joa.12384) (DSPI, single cranium).
- The `BENCH_TEST_PROTOCOL.md` reference for in vivo pig palatal strain (PMC2497442) is [Liu, Shcherbatyy & Perkins 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2497442/). It reports strains mostly <100 microstrain at 2-8 kPa chewing pressures. Check that the protocol's quoted values match this source.

## Needs Wes

1. Decide whether Paper 1 goes the Registered Report route ([RSOS](https://royalsocietypublishing.org/rsos/registered-reports)). If yes, testing must wait for Stage 1 in-principle acceptance.
2. Approve outreach to Liu/Herring, Bright/Rayfield and Moazen (draft templates exist in `AUTHOR_REQUEST_DRAFTS.md`).
3. Get institutional access or author copies of paywalled full texts so that UNVERIFIED values can be confirmed:
   - [Bright & Gröning 2011](https://doi.org/10.1002/jmor.10991)
   - [Gröning 2009](https://doi.org/10.1016/j.jbiomech.2009.03.025) and [Gröning 2012](https://doi.org/10.1016/j.jbiomech.2012.02.009)
   - [Szwedowski 2010](https://doi.org/10.1097/SCS.0b013e3181c50f64) and [Szwedowski 2011](https://doi.org/10.1243/09544119JEIM786)
   - [Boryor 2010](https://doi.org/10.1016/j.jbiomech.2010.02.002) and [Provatidis 2006](https://doi.org/10.1243/09544119JEIM92)
   - [MacLeod 2016](https://doi.org/10.1115/1.4034653)
   - [Sato 1998](https://doi.org/10.1177/00220345980770020701) (exact no-resorption thresholds)
   - [Lanyon & Rubin 1984](https://doi.org/10.1016/0021-9290(84)90003-4)
4. Approve the DATA_SOURCES A5 attribution fix.
5. Decide whether the M0 homogeneous model stays as a "known-bad" control. It is expected to fail on magnitude ([Bright & Rayfield 2011](https://doi.org/10.1111/j.1469-7580.2011.01408.x)), which is useful as a falsification demonstration.
6. Licensing: if Open-Full-Jaw geometry is used for Paper 2, CC BY-NC-SA 4.0 (per TEAM_BRIEF; not re-checked here) constrains commercial use and requires share-alike.

## Limitations of this review

- Not a systematic review. There is no PRISMA log, and searches used a small set of query phrasings. "Not found" statements are search-limited.
- Many entries are abstract-only (ABS). Numeric details beyond the abstract are marked UNVERIFIED.
- Several 2026 items (Melnyk, Lazarev, Liu 2026, Xiao, McFarland) are very recent and were read from abstracts or landing pages.
- Journal scope and fit judgments are opinion, except where a cited policy page is given.
