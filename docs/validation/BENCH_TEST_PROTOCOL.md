# Ex vivo pig-palate bench test: protocol for validating the craniofea strain field

Status: DRAFT protocol, written 2026-10-17 on branch `validation-prep`. Nothing has been built or tested yet. Every number from the scripts below is a **verification** (math/code) result. Real-world **validation** data do not exist until Wes runs the test. Prices were checked on 2026-10-17 at the URLs given. Anything not checked is marked UNVERIFIED.

Files:

| File | Purpose |
|---|---|
| `validation/bench/predict_strains.py` → `predicted_strains.json` | Order-of-magnitude strain predictions (analytic plate, FE bench analog, craniofea template) |
| `validation/bench/predict_pad_sensitivity.py` → `pad_sensitivity.json` | Sensitivity to pad pressure distribution, misalignment, pot compliance, thickness, span |
| `validation/bench/bench_fe_model.py` | Specimen-specific FE model from CT plus virtual gauges (`--selftest` passes) |
| `validation/bench/analyze_bench.py` | Pre-registered analysis and acceptance test (`--demo` runs on synthetic data) |
| `validation/bench/firmware/bench_daq/bench_daq.ino`, `host_logger.py` | DAQ firmware outline and host logger (untested) |
| `validation/bench/rig.scad` | OpenSCAD source for the rig parts (not rendered here because openscad was not installed) |
| `docs/validation/bench_bom.csv` | Bill of materials with URLs and prices |

---

## 0. Why this test, and what it can and cannot show

PR #1's headline is that a 2 kPa resting tongue posture produces no density change, while chewing provides about 98% of the stimulus. That claim depends on the FE strain magnitude per unit palatal pressure. This bench test checks one link in that chain: **whether the craniofea pipeline (CT → mesh → material → BCs → solve) predicts the measured surface strain of a real maxillary palate under a known palatal pressure.** That covers both the pattern and the magnitude.

It does **not** validate:

- the remodeling law;
- human adolescent material properties;
- in vivo boundary conditions (sutures attached to the cranium, muscle forces);
- the 2 kPa load case directly. At 2 kPa, strains are at the gauge noise floor (section 2). That case is covered only through demonstrated linearity, plus an optional averaged exploratory run (E3).

Precedent: Bright & Rayfield ran an ex vivo pig-cranium strain-gauge validation of FE. Their homogeneous 12.5 GPa model was about an order of magnitude too stiff, while strain orientations matched within 10 degrees ([Bright & Rayfield 2011, J Anat](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1469-7580.2011.01408.x)). A magnitude failure is therefore a realistic outcome. The acceptance criteria are written so that such a failure is reported, not explained away.

---

## 1. Hypotheses and acceptance criteria (frozen BEFORE testing)

Primary quantity: the **sensitivity** S of each gauge grid, in microstrain per kPa of cup pressure. S is the slope of strain against pressure over the load steps. Using slopes removes zero drift and makes the result independent of the absolute load.

Models are pre-registered. All are frozen, and their predictions hashed and committed by Wes, before any strain data are viewed:

- **M1 (primary, context of use):** the craniofea layered default, as used in PR #1: cortical 15 GPa, cancellous 1 GPa, suture 10 MPa (`bench_fe_model.py`, `material.model = "M1"`).
- **M2 (secondary):** CT-density-mapped, E = a·ρ^b with ρ from an HA-phantom-calibrated scan. a and b are fixed in the spec file before testing.
- **M0 (reference):** homogeneous E = 12.5 GPa, ν = 0.35, matching Bright & Rayfield.

**H1 (pattern + magnitude):** for M1, pooled over all valid grids and specimens, the measured S agrees with the predicted S. "Agrees" means all of the following (`ACCEPT` in `analyze_bench.py`):

| Criterion | Threshold |
|---|---|
| R² of S_meas vs S_pred | ≥ 0.80 |
| OLS slope (measured on predicted), point estimate | 0.80–1.25 |
| Slope 95% CI (specimen-cluster bootstrap) | inside 0.75–1.33 |
| abs(intercept) | ≤ 10% of max abs(S_meas) |
| NRMSE = RMSE / max abs(S_meas) | ≤ 0.20 |
| Grids within ±30% of prediction (or ±0.03 µε/kPa when small) | ≥ 70% |
| Specimens with their own slope in 0.67–1.5 | ≥ 3 of 4 |
| Rosette principal-angle error, where abs(ε1) > 2·abs(ε2) (secondary) | ≤ 15° |

**Falsification:** if the M1 pooled slope falls outside 0.5–2.0, the claim that craniofea predicts palatal strain magnitude within a factor of 2 is rejected. PR #1's 2 kPa conclusion would then need an explicit stiffness-uncertainty caveat. Any scale error found here is reported as a factor. It is never "fixed" by tuning E after unblinding.

**H2 (linearity, needed to extrapolate to 2 kPa):** each valid grid's strain vs pressure is linear over 10–70 kPa. The fit residual rms must be ≤ max(2.5 µε, 3% of peak), and the zero return after unload must be ≤ 5 µε.

These thresholds sit in the range of published bone FE validations, adjusted for a DIY rig. Examples include whole-bone strain-gauge validations reporting R² of about 0.8–0.95 with slopes near 1 ([d-nb.info thesis](https://d-nb.info/1132410584/34); [PMC9085891](https://pmc.ncbi.nlm.nih.gov/articles/PMC9085891/); [Wright State mandible thesis](https://corescholar.libraries.wright.edu/cgi/viewcontent.cgi?article=1256&context=bie)). The exact numbers here are a judgment call made in advance (Needs Wes, decision D1).

QC exclusions are applied before comparison and logged: residual rms over the linearity threshold, shunt-cal error > 2%, debonded gauge, or a gauge within 3 mm of a pot edge or suture as installed. R² is deliberately not used for QC because it would preferentially discard small but valid signals.

---

## 2. Predicted strain magnitudes and recommended load range (verification)

All numbers are linear-elastic and given per newton of total pad force, so they scale with F (`predicted_strains.json`).

**Analytic clamped strip.** Palate idealised as a plate clamped at the two alveolar pots, E = 12.5 GPa, 40 mm wide strip, uniform load. Units µε/N, edge / midspan:

| Span \ thickness | 2 mm | 3 mm | 4 mm | 6 mm |
|---|---|---|---|---|
| 25 mm | 5.5 / 2.7 | 2.4 / 1.2 | 1.4 / 0.7 | 0.6 / 0.3 |
| 35 mm | 7.7 / 3.8 | 3.4 / 1.7 | 1.9 / 1.0 | 0.9 / 0.4 |

**FE bench analog** (craniofea `ReferenceSolver`, tet4). Plate 45 × 50 mm with 5 mm rigid pot bands; uniform pressure on a 25 × 40 mm (10 cm²) patch. Peak abs(strain): t3 3.0, t4 1.8, t6 0.9 µε/N. A refined t4 mesh (218k tets) gives a peak of 2.1 and a midspan of 1.1 µε/N.

Code check: under full-span load the FE and analytic edge strains are 2.3 vs 2.7 (t3), 1.4 vs 1.5 (t4) and 0.7 vs 0.7 (t6), so the analytic and FE estimates agree within about 10–15%.

Pattern: the oral midspan is compressive, and the oral surface near the pots is tensile.

**craniofea template** (human adolescent maxilla; palate area 1603 mm², so 2 kPa = 3.21 N). With the repo's fixed BCs: abs-max 8.1, p90 0.82, median 0.34 µε/N, suture max 18.9 µε/N. With pot-like BCs: abs-max 5.3, p90 0.80, median 0.37 µε/N. So 2 kPa gives about 1 µε median, which is at the noise floor.

**Scaled to load** (t4 bench analog, midspan / peak, µε):

| Load | Midspan | Peak |
|---|---|---|
| 2 N (≈ 2 kPa on 10 cm²) | 2.1 | 3.7 |
| 20 N | 21 | 37 |
| 50 N | 52 | 92 |
| 100 N | 104 | 185 |
| 200 N | 207 | 369 |

**Noise and damage limits:**

- Expected gauge noise is about 0.1 µε from the ADC alone. That follows from 50 nV rms at gain 128 and 10 SPS ([NAU7802 datasheet](https://cdn.sparkfun.com/assets/e/c/8/7/6/NAU7802_Data_Sheet_V1.7.pdf)) with about 0.5 µV/µε bridge output at 1 V excitation.
- In practice, wet bone shows about 1–2 µε noise and drift. Bright & Rayfield measured noise SD 2.0 µε and drift 11.6 µε per 30 min.
- Damage threshold: trabecular apparent yield strain is about 0.78% in tension ([Morgan & Keaveny 2001](https://www.sciencedirect.com/science/article/pii/S0021929098000578)), and tissue-level yield is about 0.41–0.83% ([QUB](https://pureadmin.qub.ac.uk/ws/portalfiles/portal/140844840/1_s2.0_S1751616116300728_main.pdf)). Peak strains are kept below 1000 µε, at least 4× below yield.
- For context, in vivo pig palatal strains during chewing are about 30–45 µε ([PMC2497442](https://pmc.ncbi.nlm.nih.gov/articles/PMC2497442/)). Intermaxillary sutural strains reach about 1100 µε ([Rafferty, Herring & Marshall 2003](https://www.tamug.edu/marshall/Pubs_repository/Rafferty_Herring_and_Marshall-2003.pdf)).

**Recommended load range: 10–70 kPa cup pressure, about 10–70 N on a 10 cm² aperture.**

- Steps: 0, 10, 20, 35, 50, 70, 50, 35, 20, 10, 0 kPa, with 30 s plateaus. Run 3 trials per specimen.
- At 20 N the expected gauge strains are about 15–40 µε, giving SNR ≥ 10 against about 1.5 µε noise.
- At 70 N the predicted peak is about 130 µε for 4 mm bone and about 230 µε for 3 mm bone. Both are well below 1000 µε.
- The 70 kPa ceiling comes from the MPRLS range: 0–25 psi absolute, so about 70 kPa gauge ([Adafruit 3965](https://www.adafruit.com/product/3965)).
- Optional extension to about 100 N: close the stopcock to the MPRLS and compute p from load-cell F / A_eff. Do this only if the specimen-specific FE predicts a peak under 1000 µε and the measured gauges are still under 300 µε at 70 kPa.

Thin (2–3 mm) specimens nearly double the strain per newton, so Wes sets each specimen's ceiling from its own frozen FE prediction (Needs Wes, D2).

**Exploratory E3, physiological 2 kPa:** 50 cycles of 0 ↔ 2 kPa at 0.05 Hz, synchronously averaged. Noise falls by a factor of about √50 ≈ 7, to roughly 0.2–0.3 µε. That makes about 1–4 µε signals detectable and tests linearity down to the PR #1 load. This result is reported but is not part of acceptance.

**Sensitivity results** (`pad_sensitivity.json`; % change in gauge strain vs uniform t4 base):

| Change | Effect | How the design handles it |
|---|---|---|
| Parabolic pad pressure | +34 to +44% | Hydrostatic bladder gives uniform p by physics |
| Rim-concentrated pressure (stiff pad) | −40 to −49% | Hydrostatic bladder gives uniform p by physics |
| Pad misaligned 3 mm transverse | ≤ 7% | Alignment jig |
| Pad misaligned 5 mm antero-posterior | about 3% | Alignment jig |
| Compliant pot (only outer 2 mm fixed) | ±32–34% | Pot modeled from CT; Bondo E measured; sensitivity reported |
| Thickness 3 mm / 5 mm (vs 4) | +71% / −35% | Taken from CT, specimen-specific |
| Span 30 / 40 mm (vs 35) | −25 to −33% / +27 to +30% | Taken from CT, specimen-specific |

The pressure distribution and pot boundary condition are the two dominant controllable uncertainties. That finding drove the loading design: a hydrostatic bladder pad (section 4) and a modeled pot.

---

## 3. Specimens: sourcing, preparation, storage, potting

**Sourcing.** Buy 5 fresh pig heads from a butcher or abattoir: 4 for testing and 1 practice. They must be **unsplit**; many butchers saw heads in half through the midline, which destroys the palate. Ask for heads from market-weight pigs from the same supplier, collected within 24 h of slaughter.

Price: in the UK a half head is about £8.99 ([Broughs](https://www.broughs.com/p/Pigs_Head_sold_per_half_3kg_each/baec1f619c3448779b46ecc1337db77b/)). The US price is UNVERIFIED (assumed $15 each). Record supplier, date and approximate age or weight.

Reference anatomy: the pig bony palate is about 20.8 cm long and 6.7 cm at its widest ([JVAS](https://jvas.in/article/gross-and-morphometrical-studies-on-the-bony-palate-of-the-domestic-pig-sus-scrofa-domesticus/)). The porcine palatine process is porous, with BV/TV about 0.47 ([PMC9192817](https://pmc.ncbi.nlm.nih.gov/articles/PMC9192817/)). Pig palatal thickness is UNVERIFIED and will be measured from CT. Human palatal thickness is 2.8–7.0 mm ([HU Revista](https://periodicos.ufjf.br/index.php/hurevista/article/download/30877/21662)).

**Preparation** (wear cut-resistant gloves and nitrile over them; see section 9):

1. Remove the mandible and soft tissue. Saw transversely to isolate a segment of about 50 mm antero-posterior, centred on the premolar–first molar region where the palate is flat. Saw horizontally about 15 mm above the nasal floor to remove the snout, turbinates and nasal bones.
2. Cut the vomer and septum flush with the nasal floor. Strip the palatal mucosa from the oral surface. Palatal rugae are imprinted on pig palatal bone ([Laboratory Animals 2012](https://journals.sagepub.com/doi/10.1258/la.2012.012062)), so expect a ridged oral surface. Place oral gauges between ridges.
3. Trim the teeth flush with the alveolar crest to fit the CT bore (≤ 85 mm diameter including pots for a 9 cm bore Skyscan 1276, [Cornell BRC](https://www.biotech.cornell.edu/core-facilities-brc/services/x-ray-micro-ct)).
4. Keep the bone wet at all times with 0.9% saline made from non-iodized salt and distilled water. PBS tablets cost about $100 per 100 ([Fisher](https://www.fishersci.com/shop/products/molecular-probes-pbs-phosphate-buffered-saline-tablets/003002)), which is not justified.

**Storage.** Wrap in saline-soaked gauze, seal in a double zip bag, and freeze at −20 °C. Allow at most 2 freeze–thaw cycles in total: CT → test. Thaw at 4 °C for 12–24 h, then 1 h at room temperature.

Freezing at −20 °C has small effects on cortical bone elastic properties in the literature ([PubMed 19081098](https://pubmed.ncbi.nlm.nih.gov/19081098/); [Acta Orthop](https://actaorthop.org/actao/article/download/18964/22774/65247); [HAL review](https://hal.science/hal-04668000/document)). It is still a confound, so every specimen gets the same history. Bright & Rayfield kept specimens in 50:50 glycerine–water; that is not used here, to keep hydration close to physiological.

**Potting.**

- Each alveolar process (left and right) is set in a 3D-printed open cup (`rig.scad`, part `pot`) filled with Bondo body filler. Bondo is $18.98 per 28 oz ([Home Depot](https://www.homedepot.com/p/Bondo-1-qt-28-oz-Body-Filler-PN0262/202077782)). Self-cure PMMA is the alternative if exotherm is a concern.
- Add two 2 mm steel cross-pins per side through the pot wall into the alveolar bone to prevent slip.
- The free span between the pot walls is set by a printed spacer bar so the cups are parallel and level.
- Leave at least 3 mm of bare bone between the pot edge and any gauge.
- Press 4 fiducial 1/16 in chrome-steel balls ($1.24 each, [Bearings Direct](https://bearingsdirect.com/1-16-loose-chrome-ball-bearing/)) into each pot wall. They define the specimen coordinate frame on CT and on the bench.
- Cast a 10 × 10 × 80 mm Bondo bar from the same mix. Measure its E by 3-point bending using the rig's load cell and a dial indicator: E = F L³ / (48 δ I).
- In the FE model, the pot is either rigid (bone nodes inside the pot volume fixed) or elastic (pot segmented from CT with the measured E). Both are run; the elastic pot is primary for M1 and M2.

---

## 4. Loading rig

**Principle: a hydrostatic bladder.** A rigid 3D-printed cup has a 25 × 40 mm aperture with latex dental dam clamped across it. The cup is filled with water and held 1–2 mm off the oral surface, so only the membrane touches bone.

Pressure inside the cup is uniform by hydrostatics. The membrane's tension is internal to the cup–fluid–membrane system, so the inline load cell reads exactly the bone contact force F. The effective contact area is A_eff = F/p, measured every trial and passed to the FE model as the pad footprint area. This removes the ±35–50% pad-distribution uncertainty found in section 2.

Fallback: a cast silicone pad (Mold Star 15, [Smooth-On](https://shop.smooth-on.com/mold-star-15-slow)) with the full distribution uncertainty carried in the error budget.

```
            3/8"-16 threaded rods (x2), nuts above+below crosshead
   ||==========[ crosshead ]==========||
   ||              | M6 screw (sets cup height / standoff)
   ||          [ S-type load cell 50 kg ]
   ||          [ adapter + 8 mm ball seat ]
   ||          [ pressure cup, aperture down ]==luer== stopcock ==+== MPRLS (0-25 psi abs)
   ||    ____   ~~ latex dam ~~   ____    ||                      +== 20 mL syringe in screw press
   ||   | pot |===== palate =====| pot |  ||                      +== bleed / fill
   ||   | cup |  (nasal gauges)  | cup |  ||   oral surface up, 15 mm clearance under nasal floor
   ||===[          base plate          ]==||   pots on 6 mm dowels = repeatable position
```

**Parts** (`rig.scad`, select with `part = "..."`):

- `base`: plywood or printed base with rod holes and dowel holes.
- `pot` ×2.
- `crosshead`.
- `cup`: luer port, bleed hole and M3 flange holes.
- `clamp_ring`: clamps the dam.
- `adapter`.
- `jig`: placeholder for the per-specimen gauge template generated from CT.

Hardware:

- Threaded rod 3/8 in × 6 ft, $10.95 ([Home Depot](https://www.homedepot.com/b/Hardware-Metal-Stock-Metal-Rods/N-5yc1vZ2fkos9n)).
- Load cell: Phidgets 50 kg S-type, $40 ([Phidgets](https://www.phidgets.com/?prodid=1319)), read by its own NAU7802.
- Pressure source: a 20 mL luer syringe ([Filtrous](https://filtrous.com/products/luer-lock-syringe-sterile-20ml-50-unit)) driven by a printed M8 screw press, with 3-way stopcocks ([JorVet](https://jorvet.com/shop/jorvet-3-way-stopcock/)).
- A 2–3 mL air pocket in the syringe gives compliance for smooth pressure control. The cup itself must be bubble-free; bleed it while inverted.

**Alignment and standoff:**

1. Lower the cup onto a 1.5 mm printed shim placed on the bone. Lock the crosshead nuts, then remove the shim.
2. Centre the cup over the planned aperture using the printed alignment jig that registers to the pot fiducials. The target is ±1 mm, which costs ≤ 7% per section 2.
3. The ball seat between the load cell and cup keeps the force axial.

**Rig verification before any specimen:**

1. Dead-weight calibration of the load cell: 0.5–10 kg kitchen weights on a 1 g-resolution scale.
2. Calibration of A_eff against a flat rigid plate on a second scale.
3. Leak test: hold 70 kPa for 5 min with < 2% drop.
4. Compliance test.

---

## 5. Gauges, bridges, DAQ

**Gauge choice.** Use unstacked foil gauges with 1.5 mm (0.062 in) grids and STC 06. Stacked rosettes are too stiff and thick for bone; unstacked is preferred ([ORS 2018 in vivo bone loading handout](https://www.ors.org/wp-content/uploads/2018/03/InVivo_Bone_Loading_Handout_ORS2018.pdf)).

- Rosette: CEA-06-062URA-350, $223.93 per 5.
- Uniaxial: EA-06-062AQA-350, $44.09 per 5 ([Micro-Measurements gauges](https://shop.micro-measurements.com/my_cart_item_list.php?v_cat_code=STRESS_ANALYSIS_STRAIN_GAGE)).
- Bright & Rayfield used C2A-06-062LR-350 (price not found).

Per specimen:

- 1 rosette on the nasal floor at midspan (opposite the cup);
- 2 uniaxial gauges, transverse, on the oral surface 3–5 mm outside the pot-side aperture edge (tensile zone) and on the nasal surface near the pot (compressive zone);
- 1 optional exploratory uniaxial gauge across the midpalatal suture (excluded from acceptance);
- 1 dummy gauge on an unloaded bone offcut from the same specimen, in the same saline bath, for temperature compensation.

Sites are chosen from the frozen FE as places with low strain gradient (change < 10% over ±0.5 mm), at least 3 mm from sutures, pot edges and foramina. Total: 9 bridge channels plus the load cell.

**Surface preparation and bonding** (Micro-Measurements accessories, [MM shop](https://shop.micro-measurements.com/my_cart_item_list.php?v_cat_code=ACCESSORIES); method after Bright & Rayfield):

1. Remove periosteum with a scalpel and lightly abrade with 400-grit paper or pumice over 6 × 6 mm.
2. Degrease with isopropanol. Apply Conditioner A ($9.97), then Neutralizer 5A ($9.97).
3. Dry locally with a stream of warm air for 1–2 min while the surrounding bone stays covered with wet gauze.
4. Bond with M-Bond 200 ($63.07), holding thumb pressure through PTFE film for 1 min.
5. Solder 134-AWP jumpers ($37.98) to CPF-75C terminals ($33.81), then 326-DFV 3-conductor leads ($59.63).
6. Cover the grid, tabs and terminal with 3140 RTV ($55.89), a non-corrosive silicone. Never use acetoxy (vinegar-smell) silicone. Cure 24 h at 4 °C under wet gauze, away from the grid.
7. Check each gauge for 350 ± 1 Ω and for > 1 GΩ to the bone (use a saline-wetted probe). Replace it if it fails.

**Bridge.**

- Each channel is a 3-wire quarter bridge ([TML bridge list](https://www.tml.jp/e/knowledge/strain_gauge/bridge_list.html)): active gauge, 350 Ω 0.1% completion, and a 1 kΩ/1 kΩ 0.1% ratio arm.
- The bridge is fed from the NAU7802 AVDD (LDO 3.0 V) through 2 × 510 Ω dropping resistors, so the bridge sees about 1 V. Gauge dissipation is then about 0.7 mW, within the low-power range Micro-Measurements recommends for poor heat-sink substrates ([TN-502](https://intertechnology.com/Vishay/pdfs/TechNotes_TechTips/TN-502.pdf)).
- Signal: about 0.5 µV/µε at GF 2.0.
- NAU7802 settings: gain 128, 10 SPS, channel 1, internal AFE offset calibration after warm-up ([datasheet](https://cdn.sparkfun.com/assets/e/c/8/7/6/NAU7802_Data_Sheet_V1.7.pdf)).
- Common-mode caveat (UNVERIFIED in hardware): the bridge common-mode voltage of about 1.5 V is at the edge of the datasheet's input range at high gain. Verify linearity with the shunt-cal steps. If they are nonlinear, raise the LDO to 3.3 V and rebalance.

**Temperature.** Bone CTE is about 27.5e-6/°C ([Eng Med 1987](https://sage.cnpereading.com/doi/10.1243/EMED_JOUR_1987_016_036_02)), while the STC 06 match is 10.8e-6/°C. The mismatch gives about 17 µε/°C apparent strain.

Countermeasures:

- log temperature with TMP117 sensors ([Adafruit 4821](https://www.adafruit.com/product/4821));
- subtract the dummy gauge in software;
- keep the saline at room temperature and keep each load ramp under 5 min;
- the slope-vs-load analysis rejects slow drift.

**Shunt calibration.** Apply a shunt across the active gauge at the gauge end of the leads ([TN-514](https://www.intertechnology.com/Vishay/pdfs/TechNotes_TechTips/TN-514.pdf)). For 350 Ω and GF 2.0:

| Shunt | Simulated strain |
|---|---|
| 348 kΩ | −502 µε |
| 174 kΩ | −1004 µε |
| 69.8 kΩ | −2495 µε |

(`host_logger.py shunt`.) This calibrates excitation, gain and lead loss per channel. Do it before and after each specimen; a > 2% change fails QC.

**DAQ hardware:**

- 10 × SparkFun Qwiic Scale NAU7802 at $5.65 each for 10+ ([SparkFun](https://www.sparkfun.com/sparkfun-qwiic-scale-nau7802.html)).
- All share address 0x2A, so they sit behind 2 × PCA9548 muxes at 0x70/0x71 ($6.95 each, [Adafruit 5626](https://www.adafruit.com/product/5626)).
- QT Py RP2040 ($9.95, [Adafruit 4900](https://www.adafruit.com/product/4900)).
- MPRLS and TMP117 on the main bus.

**Firmware** (`bench_daq.ino`, outline only):

- configure each channel through the mux;
- stream raw counts at about 10 Hz frames;
- serial commands: Z (zero), C (AFE calibration), S/X (start/stop), R (raw).

Raw counts are archived and conversion is done on the host. Event markers (load-step changes) are written by the host. Warm up for 30 min before testing.

---

## 6. CT plan, segmentation, meshing, gauge mapping

**Sequence.** CT comes first and gauges second, so the FE prediction is truly blind:

1. Pot all 4 specimens.
2. Batch-scan them in one session, without gauges, wrapped in wet gauze in sealed tubes.
3. Build the FE models and choose gauge sites.
4. Freeze the predictions, commit the hash, and print the placement jigs.
5. Install gauges, test, and photograph.
6. Optionally run a quick low-resolution post-test scan to confirm gauge positions.

**Options and cost:**

| Option | Cost | Notes |
|---|---|---|
| Collaborator's micro-CT (preferred) | $0–internal rate | Ask a dental, vet or biomech lab |
| Cornell BRC Skyscan 1276 | $180/h external academic, $275/h commercial | 9 cm bore, 2.8–80 µm voxels ([Cornell](https://www.biotech.cornell.edu/core-facilities-brc/services/x-ray-micro-ct)) |
| Oregon State | external commercial minimum $900 (≤ 4 h) + $681 personnel; academic $300 | [OSU fees](https://microct.oregonstate.edu/user-fees) |
| HSS | $100–200/h | [HSS](https://www.hss.edu/research/core-facilities/molecular-histopathology-core/services/mct-core) |
| Yale | $283/h internal | [Yale](https://research.yale.edu/cores/yale-cvrc-micro-ct-imaging-core) |
| Fred Hutch | $266.20 external | [Fred Hutch](https://www.fredhutch.org/en/research/shared-resources/core-facilities/preclinical-imaging/rates-scheduling.html) |
| Dental or vet CBCT | UNVERIFIED cost | Voxel about 0.1–0.2 mm; acceptable for geometry, poor for density (M2) |
| Clinical CT | — | 0.5 mm is too coarse for a 3–4 mm palate (Bright & Rayfield used 0.55 mm pixels and 2 mm slices) |

The budget assumes 3 h at $180/h ($540), for example two specimens per scan.

**Scan parameters** (to confirm with the facility):

- 70–90 kVp, 0.5 mm Al or 0.1 mm Cu filter;
- 40–60 µm isotropic voxels, which give at least 50 voxels across 3 mm;
- 0.4° rotation step with 2–3 frame averages;
- an HA calibration phantom in the field (required for M2);
- the Bondo bar in the same scan.

**Segmentation** (3D Slicer, free):

- bone: global threshold at half-maximum between bone and saline peaks, keep the largest island, fill internal cavities only for M1 and M0 (M2 keeps density);
- pot resin and steel balls as separate segments;
- record the fiducial ball centroids;
- export STL in CT millimetres;
- Sensitivity: run threshold ±10%. A one-voxel error on 3 mm is about ±2–3% thickness, so about ±5% strain because strain scales with 1/t².

**Mesh** (`bench_fe_model.py`):

- fTetWild through `craniofea.mesh.remesh.remesh_surface_ftetwild`, or gmsh;
- target size 0.6 mm, with at least 4 tets through the thickness;
- convergence check at 0.8 / 0.6 / 0.45 mm: virtual-gauge change < 5%, otherwise refine.

Linear tet4 elements underestimate bending strain; the bench analog's coarse→refined change was 1.8 → 2.1 µε/N peak, about 15%. Use `--solver calculix` for meshes over about 150k nodes if CalculiX is available. Within 2 CPUs and 7 GB, the reference solver handled 218k tets.

**Labels:**

- fixed = bone nodes inside each pot half-space or box (rigid), or a tied elastic pot;
- pad = oral faces inside the cup aperture projection minus a 1 mm rim band, scaled so the area matches the measured A_eff;
- load = 1 kPa;
- output = µε/kPa per grid (and per N).

Status: the M0 path is tested; the M1 element-set labeling (cortical shell / cancellous / suture) and the M2 density mapping are written as stubs and must be completed and tested before predictions are frozen (task T8 in Needs Wes, or a follow-up agent task). `--selftest` reproduces the direct-FE bench analog: nasal midspan 0.75 vs 0.77 µε/N, oral edge 0.94 vs 0.94 µε/N.

**Gauge mapping:**

- The placement jig is designed from the CT surface: a window at the planned site, axis notches, and pins seating in the pot fiducials. Expected placement accuracy is ±0.5 mm and ±3° (UNVERIFIED until tried on the practice specimen).
- After bonding, photograph each gauge with a mm scale and measure caliper distances to 3 fiducial balls. The as-installed center is trilaterated in CT coordinates.
- Predictions are re-evaluated at the as-installed location and angle with the frozen model. This is the only post-test model input allowed.

---

## 7. Analysis plan, statistics, uncertainty budget

**Processing** (`analyze_bench.py`):

1. Convert counts to µε using the shunt calibration.
2. Subtract the dummy-gauge channel.
3. Remove drift linearly between pre- and post-trial zeros.
4. For each step, take the plateau mean (last 20 s of each 30 s step).
5. Compute per-trial OLS slope S, intercept and residual rms.
6. Average the 3 trials per grid. Report the trial-to-trial CV (target < 5%).
7. Rosettes: compute principal strains and angle; compare angle and magnitude separately.

**Agreement:**

- pooled OLS of S_meas on S_pred for each model (M1 primary);
- 95% CI from a specimen-cluster bootstrap (4000 resamples), because grids within a specimen are correlated;
- R², RMSE, NRMSE, and Bland–Altman on log(S_meas/S_pred);
- the fraction of grids within ±30%;
- slope per specimen.

`--demo` (synthetic data: 4 specimens × 9 grids, model 15% too stiff, 1.5 µε noise) recovers slope 1.15 with CI 1.12–1.20 and passes all criteria. This is a code check only, not evidence.

With n ≈ 36 grids and 4 clusters the slope CI is driven by specimen-to-specimen variation. Fewer than 3 valid specimens means the test is **inconclusive**, not passed.

**Uncertainty budget** for measured vs predicted S, 1σ, per grid. Many values are estimates and are marked accordingly:

| Source | Estimate | Basis |
|---|---|---|
| Gauge factor | ±1% | MM gauge factor tolerance (typical, UNVERIFIED for these lots) |
| Shunt-cal / ADC chain | ±0.5% | 0.1% resistors; shunt repeated |
| Pressure p (MPRLS) | ±1.5% FS ≈ ±2.6 kPa abs; reduced by zeroing at atmosphere | UNVERIFIED datasheet accuracy |
| Load cell (for A_eff) | ±0.5% | C3 class; dead-weight calibrated |
| Gauge reinforcement and transverse sensitivity | ±2% | Polyimide grid on ≥ 3 mm bone, small |
| Noise / drift on slope | ±0.1 µε/kPa → about ±5% at typical 0.5–1.5 µε/kPa | Bright & Rayfield 2 µε SD; 5 plateaus × 3 trials |
| Temperature (residual after dummy) | ≤ 1 µε per ramp → ≤ 2% | 17 µε/°C × ≤ 0.05 °C in 5 min |
| Gauge position ±0.5 mm × FE gradient | ±3–5% | Sites chosen where gradient < 10% per mm |
| Gauge angle ±3° | ±1–3% (uniaxial); angle error itself for rosettes | |
| Viscoelastic / hydration change | ±3% | Short plateaus; trial-to-trial CV tracked |
| **Measured S, combined** | **about ±8–10%** | root-sum-square |
| FE: geometry / threshold | ±5% | Section 6 |
| FE: mesh discretization | ±5–15% | Convergence study |
| FE: pad footprint (bladder) | ±5% | A_eff measured; 1 mm rim band |
| FE: pot BC | ±10–30% | Elastic vs rigid pot run both ways |
| FE: material (E) | factor 2–10 possible | This is what the test measures; not "budgeted away" |

With the controllable terms at about ±15–35%, a ±30% per-grid band and a 0.75–1.33 pooled-slope band are achievable if the material model is right. They are tight enough to detect the order-of-magnitude stiffness errors seen by Bright & Rayfield.

**Reporting:** all results, including failed criteria and every exclusion, go in `docs/validation/BENCH_TEST_RESULTS.md` (future). The raw logs, frozen predictions and their hash are archived.

---

## 8. Bill of materials and cost

Full list with URLs: `docs/validation/bench_bom.csv`. Subtotals in USD, shipping and tax excluded:

| Category | Subtotal |
|---|---|
| Gauges and installation supplies | $659.08 |
| Electronics (ADCs, muxes, MCU, load cell, MPRLS, TMP117, resistors) | $255.40 |
| Rig (threaded rod, dental dam, stopcocks, syringes) | $67.28 |
| Potting and fiducials | $38.82 |
| Specimens (5 heads, price UNVERIFIED) | $75.00 |
| Safety / consumables (partly UNVERIFIED) | $54.56 |
| **Total excluding CT** | **$1,150.14** |
| Micro-CT, 3 h at $180/h | $540.00 |
| **Total with CT at the academic external rate** | **$1,690.14** |
| Total with CT at the $275/h commercial rate | $1,975.14 |

Getting under $1,500 requires either:

- (a) a collaborator scanning free or at an internal rate of about $100/h, giving about $1,150–1,450; or
- (b) a 2 h batched scan ($360) plus lean substitutions: generic 3-conductor wire instead of 326-DFV (−$60), and pharmacy swabs and gauze (−$33). That gives about $1,420.

Filament, hookup wire, solder, 510 Ω resistors, screws and plywood are assumed to be on hand.

---

## 9. Safety, ethics and step-by-step checklist

**Ethics.** The tissue is a food-supply by-product bought from a butcher. No animal is killed for the study, so no IACUC approval is normally required for a private builder (UNVERIFIED for any institution Wes collaborates with; the CT facility may have its own tissue policy, so ask). Do not use heads from animals of unknown disease status.

**Biosafety.** Raw pork can carry *Streptococcus suis*, which infects people through skin cuts while handling raw pork ([CDC EID](https://wwwnc.cdc.gov/eid/article/20/7/13-1594_article); [CHP Hong Kong](https://www.chp.gov.hk/en/healthtopics/content/24/3648.html)). Precautions:

- cut-resistant A4 gloves ([Zoro](https://www.zoro.com/k/a4-cut-resistant-gloves/)) under nitrile gloves ([Home Depot](https://www.homedepot.com/p/HDX-Blue-3-5-mil-Disposable-Nitrile-1-Size-Fits-Most-100-Count-24570-06/319285600)) when sawing;
- cover any cuts; use eye protection and a face shield when sawing bone;
- a dedicated tray and tools;
- wipe surfaces with 1:10 household bleach.

Keep tissue out of the food fridge, or double-bag it in a labelled box.

**Electrical.** The bridges run on USB 3.3 V only. Keep the USB host on battery or an isolated laptop supply while hands are in the saline bath.

**Sawing and Bondo.** Bondo releases styrene, so mix it outdoors or with ventilation.

**Pressure.** The cup at 70 kPa holds about 70 N; wear safety glasses in case the membrane fails.

**Disposal.** Double-bag the tissue, freeze it, and put it out on trash day as with butcher waste. Check local rules (UNVERIFIED). Neutralizer and conditioner go down the drain with water, per their SDS.

**Checklist** (wall-clock estimates for one person; active hours in parentheses):

| # | Step | Time |
|---|---|---|
| 1 | Wes approves criteria, model set and budget (D1–D4); commit nothing yet | 1 h |
| 2 | Order BOM; print rig parts (`rig.scad`); cut rods | 1–2 weeks lead (4 h) |
| 3 | Build DAQ boards and firmware; warm-up noise test with 350 Ω dummy resistors (target < 0.3 µε rms) | 2 days (8 h) |
| 4 | Rig verification: load-cell dead weights, cup leak test, A_eff calibration | 1 day (4 h) |
| 5 | Practice head: dissect, pot, bond 3 gauges, full dry run including shunt cal and 3 trials | 2 days (8 h) |
| 6 | Collect 4 heads; dissect and trim; pot with cross-pins and fiducials; cast Bondo bar; freeze | 1 day (6 h) |
| 7 | Bondo bar 3-point bend → E_pot | 1 h |
| 8 | Micro-CT batch (all 4 plus phantom and bar), transport frozen in a cooler | 0.5–1 day (3–6 h) |
| 9 | Segment, mesh, convergence check, run M0/M1/M2; choose gauge sites; print jigs | 3–4 days (12 h, compute about 1 h per specimen) |
| 10 | Freeze predictions: Wes commits spec + predicted.csv + SHA-256 | 0.5 h |
| 11 | Per specimen: thaw, install gauges (cure overnight under wet gauze), measure positions | 2 days each (4 h) |
| 12 | Per specimen: mount, warm up, shunt cal, 3 trials 0–70 kPa, E3 averaged 2 kPa, shunt cal, photos | 0.5 day each (3 h) |
| 13 | Optional post-test low-resolution scan | 1 h scanner |
| 14 | Run `analyze_bench.py`; write results; dispose of tissue | 1 day (4 h) |

Total: about 4–6 weeks calendar time and about 75–90 h of Wes's active time.

---

## Needs Wes

Purchases:

| Item | Cost | Time |
|---|---|---|
| P1. Micro-Measurements order: 1 rosette pack, 3 uniaxial packs, M-Bond 200, conditioner, neutralizer, 3140 RTV, terminals, wire | $627 (or $567 without 326-DFV) | 15 min to order; about 1–2 weeks lead (UNVERIFIED) |
| P2. Electronics: SparkFun, Adafruit, Phidgets, Mouser | $255 | 30 min |
| P3. Hardware, potting, fluidics, safety (Home Depot, eBay, JorVet, Filtrous, Bearings Direct, Zoro) | about $160 | 30 min + store trip |
| P4. 5 unsplit fresh pig heads from a butcher | about $75 (UNVERIFIED) | 1 h incl. phone call; must be "unsplit, mandible optional" |
| P5. Micro-CT time | $0 (collaborator) to $540–825 | 1–2 h of emails + 0.5–1 day at the scanner |

Physical tasks:

| Item | Time |
|---|---|
| T1. Print rig parts from `rig.scad` and check fit (openscad was not available here, so the geometry is unchecked) | 4–6 h |
| T2. Build and test the DAQ; fill in MPRLS/TMP117 code in `bench_daq.ino`; verify NAU7802 common-mode linearity with shunt cal | 8 h |
| T3. Rig verification: load cell, leak, A_eff | 4 h |
| T4. Practice specimen and gauge-bonding practice | 8 h |
| T5. Dissection, potting, Bondo bar test | 7 h |
| T6. Gauge installation and testing, 4 specimens | 28 h |
| T7. Transport specimens to CT frozen, and sign any facility tissue forms | 0.5–1 day |

| T8. Complete and test the M1 element-set labeling and M2 density mapping in `bench_fe_model.py` (an agent can do this once a real CT exists) | 4–8 h |

Decisions:

| Item | Time |
|---|---|
| D1. Approve or adjust the acceptance thresholds in section 1 before any data exist; they are then frozen | 30 min |
| D2. Per-specimen maximum pressure (default 70 kPa) and whether the 100 N extension is allowed | 10 min per specimen |
| D3. CT route: collaborator vs Cornell vs other; this sets whether the budget is under $1,500 | 30 min |
| D4. Bondo vs PMMA potting | 10 min |
| D5. Commit and push protocol, frozen predictions and hash. The subagent was told not to commit, so only Wes can create the timestamped pre-registration | 15 min |
| D6. If H1 fails for M1: publish the scale factor and caveat PR #1's 2 kPa result rather than retune. Agree to this in advance | 15 min |
