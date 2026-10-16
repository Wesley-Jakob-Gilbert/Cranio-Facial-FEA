# Methods: materials, daily loading and strain-adaptive remodeling

This document describes the model implemented in `craniofea.materials`,
`craniofea.remodel` and the study runner (`craniofea study`,
`craniofea sensitivity`). The question it explores is whether sustained
tongue-on-palate posture ("mewing") could change adolescent maxillary bone
density compared with mouth breathing (tongue low, no palatal contact). It is
a hypothesis explorer. It is not a clinical predictor, and it has not been
validated against longitudinal imaging.

Every constant below is a field of `craniofea.config.StudyConfig` and can be
set in YAML (`configs/study.yaml`, `configs/study_template.yaml`) or varied
in the sensitivity analyses.

## 1. Overview

```
Psi*_e  = Psi_e(reference habit, rho = 1)                       homeostatic setpoint
for each habit h, rho^0 = 1, step k = 0 .. K-1 (dt = T / K days):
    material(rho^k) -> one FE Problem with all load cases of h -> U_ie
    eps_ie  = sqrt(2 U_ie / E_eff,e)
    Psi_e   = (sum_i n_i eps_ie^m)^(1/m)
    S_e     = smooth(Psi_e / Psi*_e - 1)
    rho^{k+1} = clip(rho^k + dt * r(S_e))
```

## 2. Tissue materials (`craniofea/materials.py`)

Each element belongs to exactly one layer: `cortical`, `cancellous` or
`suture_mid` (canonical elem sets, see `docs/ARCHITECTURE.md`).

**Density-modulus law.** The relative density rho (1.0 = tissue at the start
of the study) scales the stiffness of the bone layers by a power law
(Carter & Hayes 1977; quadratic dependence for cancellous bone after Rice,
Cowin & Bowman 1988):

    E(rho) = E_layer * rho^n_layer,      n_cortical = 3,  n_cancellous = 2

Poisson ratios do not depend on density. Bone mass density scales linearly
with rho. Mass only enters through gravity, which is a minor load case.

**Cortical anisotropy (optional; on in `configs/study.yaml`).** The cortex is
orthotropic in local axes that follow the cortical plate. Axis 3 is the
outward surface normal (elem field `surface_normal`). Axis 1 is the anatomical
`ap` axis projected onto the tangent plane. When that projection is shorter
than 0.2, `lr` and then `si` are used instead, and at least one of the three
always qualifies. Axis 2 = axis 3 x axis 1. The local stiffness `C_loc` is
built from the nine engineering constants. It is rotated to global axes with
the engineering-strain Bond transformation, `C = T^T C_loc T` where
`eps_loc = T eps_glob`. Every Young's and shear modulus is scaled by
`rho^n_cortical`.

**Isotropic-equivalent modulus.** The stimulus needs one modulus per element.
We use the Voigt-Reuss-Hill average of C (Hill 1952), which does not depend
on the rotation and equals E exactly when the element is isotropic:

    K_V = [C11+C22+C33 + 2(C12+C13+C23)]/9,  G_V = [C11+C22+C33 - (C12+C13+C23) + 3(C44+C55+C66)]/15
    1/K_R = S11+S22+S33 + 2(S12+S13+S23),     15/G_R = 4(S11+S22+S33) - 4(S12+S13+S23) + 3(S44+S55+S66)
    K = (K_V+K_R)/2, G = (G_V+G_R)/2,         E_eff = 9KG/(3K+G)

**Sutures.** The midpalatal suture has a constant isotropic modulus and is not
remodeled. The legacy pipeline had tension-triggered softening and
ossification rules. We dropped them because their rate constants and
thresholds had no calibration source. Sutural growth is also a cell and
tissue process (proliferation, matrix deposition, fusion), and a
strain-adaptive density law does not describe it. The suture modulus is
therefore a fixed mechanical boundary condition, varied over two decades in
the sensitivity analysis.

## 3. Daily loading history (`craniofea/remodel/loading.py`)

A habit is modelled as the set of distinct mechanical events in a typical
day. Each event is an explicit FE load case built from canonical surface sets
and `mesh.frame`:

| Case | Type | Load | Daily count n_i |
|---|---|---|---|
| `rest_posture` | sustained | pressure on `palate`, mean `resting_pressure` (per habit), linear AP gradient `resting.ap_ratio` | `static_equivalent_cycles` |
| `swallow` | cyclic | peak palatal pressure `swallowing.pressure`, AP ratio `swallowing.ap_ratio` | `swallowing.cycles_per_day` |
| `chew` | cyclic | bite reaction on `occlusal`, total `mastication.force`, along +si (tilt `inclination_deg` toward +ap), posterior bias `molar_weight` | `mastication.cycles_per_day` |
| `gravity` | sustained | self-weight along -si | `static_equivalent_cycles` |

**AP pressure gradient.** Let t in [0, 1] be the normalised AP coordinate of a
face centroid over the loaded set (0 = most posterior). The pressure is

    p_f = p_mean g(t_f) / <g>_A,    g(t) = 1 + (ratio - 1) t,

where `<g>_A` is the area-weighted mean of g. The resultant force therefore
equals `p_mean x area` for any ratio. Positive pressure pushes into the bone,
so on the oral palatal surface it acts superiorly.

**Bite reaction.** The face weights are `w_f = A_f (molar_weight + (1 - molar_weight) t_f)`,
normalised to sum to 1. Each face carries `F w_f d`, split equally over its
three nodes, so the resultant is exactly `F d`.

**Habits.** `reference` is the habitual resting posture, with a small resting
pressure. It defines the setpoint. `mewing` is an elevated sustained palatal
pressure. `mouth_breathing` has zero resting palatal pressure. Swallowing and
chewing are the same in all three habits unless a habit sets
`swallowing_scale` or `mastication_scale`. Load cases with zero load are
omitted.

## 4. Daily stimulus (`craniofea/remodel/stimulus.py`)

For element e and load case i, with strain energy density U_ie:

    eps_ie = sqrt(2 U_ie / E_eff,e)

This is the exact axial strain for uniaxial stress (tested). In general it is
the strain of an isotropic element with modulus E_eff that stores the same
energy. It is the strain counterpart of the effective stress
`sigma_bar = sqrt(2 E U)` of Carter, Fyhrie & Whalen (1987) and Beaupré, Orr &
Carter (1990). Because the strain is computed with the current, density-scaled
modulus, denser bone with the same load sees a smaller stimulus. This is the
negative feedback that drives the law towards equilibrium.

The daily stimulus combines the day's events with their cycle numbers (Carter
et al. 1987; Beaupré et al. 1990):

    Psi_e = ( sum_i n_i eps_ie^m )^(1/m),        m = 4

With m = 4, magnitude counts far more than cycle number. Doubling the strain
of an event has the same effect as multiplying its cycle count by 16.

**The sustained-load weight.** A sustained load is not cyclic, so its n_i is
not defined by the formalism. We expose it as `static_equivalent_cycles`. The
default of 1 per day treats a sustained load as one load cycle. This follows
the animal evidence that static loads are weakly osteogenic or not osteogenic
at all, while a small number of dynamic cycles is enough (Rubin & Lanyon 1984;
Lanyon & Rubin 1984; reviewed by Turner 1998).

A rough upper reference is to count a load held all day as if it were cycled
at 1 Hz, which gives 86,400 cycles/day. This is not a physiological claim: a
static load produces no strain rate. It is an extreme bound on how generously
the posture could be credited. **The mewing hypothesis depends entirely on
this parameter**, so the study reports its results as a function of it
(section 7).

## 5. Remodeling law (`craniofea/remodel/law.py`)

**Site-specific setpoint.** `Psi*_e = Psi_e(reference habit, rho = 1)`. There
is no global, hand-tuned setpoint. The reference habit is in equilibrium by
construction: S = 0 exactly and rho does not change (tested). Every other
habit is judged relative to it. This removes the legacy problem where the
outcome was set by an arbitrary global SED target. The cost is that results
can only be *relative to the reference habit*. Elements with
`Psi* <= setpoint_floor x max(Psi*)` carry no information and are held fixed.

**Error, lazy zone and rate.**

    S_e = Psi_e / Psi*_e - 1
    drho/dt = c_form (S - w)   if S >  w
            = 0                if |S| <= w
            = c_res  (S + w)   if S < -w          [1/day]

This is the trilinear "dead zone" law of Beaupré et al. (1990), written for a
relative density rather than a surface apposition rate. It is equivalent to
the lazy zone used by Huiskes et al. (1987). Suture elements have zero rate.

**Time integration.** Explicit Euler with dt = T/K days, re-solving the FE
problem at every step:

    rho^{k+1} = clip(rho^k + dt drho/dt, rho_min,layer, rho_max,layer)

All load cases of one step are solved with one `Problem`, so they share one
stiffness matrix.

**Smoothing (default on).** S is averaged to the nodes, volume-weighted over
bone elements only, and back to the elements as the mean of their four nodes.
This linear filter damps the element-scale checkerboard pattern of
element-based density laws (numerical stability of remodeling schemes:
Garijo et al. 2014). It leaves a uniform S unchanged and maps S = 0 to 0, so
homeostasis is preserved exactly.

**Rate constants.** The rates are given in relative density per day per unit
error. Their scale comes from adult bone turnover: about 4 %/yr in cortical
and about 28 %/yr in trabecular bone (Parfitt 1994, tabulated in Manolagas
2000, via BioNumbers ID 109908). With an error one unit beyond the lazy zone
(S - w = 1, i.e. Psi = (2 + w) Psi* = 2.1 Psi*), the yearly density change then
equals the yearly turnover fraction:

    c_cortical = 0.04 / 365 = 1.1e-4 /day,    c_cancellous = 0.28 / 365 = 7.7e-4 /day

Adolescent turnover is higher than adult turnover, and formation may not be
symmetric with resorption. Both are open, and both rates are configurable and
varied in the sensitivity analysis.

## 6. Parameters

Confidence: **H** = directly measured value in the cited source, **M** = within
a reported range but the choice is a judgement, **L** = assumption or
extrapolation (not measured for the adolescent human maxilla).

| Parameter | Default | Range used | Source / note | Conf. |
|---|---|---|---|---|
| Cortical E (isotropic) | 15 GPa | 10-20 GPa | typical FE value for jaw cortical bone (about 8-15 GPa in maxillary FE models); no verified measurement for the adolescent maxilla | M |
| Cortical orthotropic E1/E2/E3 | 20/13/11 GPa | E1 15-30 GPa | Mandibular cortex: longitudinal 20-30 GPa and 50-100 % stiffer than the other two directions (Schwartz-Dabney & Dechow 2003). Assigning these axes to the maxilla is an assumption | L |
| Cortical G12/G13/G23 | 5/4/3.5 GPa | - | representative, not measured here | L |
| Poisson ratios (bone) | 0.30 | - | conventional | M |
| Cancellous E | 1.0 GPa | 0.3-2 GPa | typical FE value; measured mandibular trabecular apparent moduli are much lower (6.9-199.5 MPa, Lakatos et al. 2014), so this is uncertain | L |
| Density exponent n | 3 (cort.), 2 (canc.) | - | Carter & Hayes 1977; Rice, Cowin & Bowman 1988 | M |
| Suture E | 10 MPa | 1-100 MPa | FE models use about 0.7-69 MPa. Swine midpalatal suture 12.5/31.3/98.5 MPa (premaxillary/maxillary/palatine; Savoldi et al. 2018, four-point bending) | L |
| Suture nu | 0.45 | - | near-incompressible soft tissue, assumption | L |
| Mass density cort./canc./suture | 1850/1000/1100 kg/m^3 | - | representative; affects gravity only | M |
| Reference resting pressure | 200 Pa | 50-500 Pa | resting tongue pressures about 110-200 Pa at premolar/molar sites (Zeng et al., pilot study, journal not verified) and about 1 g/cm^2 = 98 Pa on the lingual surface of maxillary teeth (Thüer, Sieber & Ingervall 1999). About half of subjects had *negative* palatal vault pressure at rest (Thüer et al. 1999) | M |
| Mewing resting pressure | 2 kPa | 1-4 kPa | no measurement of deliberate posture found. 2 kPa is about 5 % of maximal tongue pressure (10-41 kPa, Hayashi et al. 2002), plausible for a sustained voluntary effort | L |
| Resting AP ratio | 2.0 | 0.5-3 | conflicting evidence: resting pressure fell from anterior to posterior upright (Zeng et al.), but another study found the reverse. Mewing instruction puts the tongue tip anteriorly | L |
| Swallowing pressure | 15 kPa | 5-30 kPa | 3-27 kPa for 5 mL water swallows (Hayashi et al. 2002). Sensor-sheet studies report peaks of about 10-16 kPa per channel (secondary sources, full texts not checked). Saliva swallows exceed water swallows (Fei et al. 2013) | M |
| Swallowing AP ratio | 1.0 | - | sensor-sheet channels have similar peaks; anterior onset is earlier (Hori et al. 2009) | M |
| Swallows per day | 600 | 230-1000 | about 585/day, range about 233-1008, adults (Lear, Flanagan & Moorrees 1965) | M |
| Bite reaction per chewing cycle | 100 N | 50-250 N | maximal molar bite force rises through adolescence and peaks at about 16 y (Roldán et al. 2016). That chewing forces are a fraction of the maximum is an assumption | L |
| Chewing cycles per day | 1500 | 200-3300 | chewing frequency 1.57 Hz (Po et al. 2011). Daily chewing time differs about 10-fold between populations, mean about 35 min/day (van Casteren et al. 2022). 1500 is about 16 min/day | L |
| Molar weighting | 3 | - | posterior bias of chewing loads, assumption | L |
| Gravity | 9.81 m/s^2 along -si | - | | H |
| Stimulus exponent m | 4 | 2-8 | Carter et al. 1987; Beaupré et al. 1990 | M |
| `static_equivalent_cycles` | 1 /day | 0-1e7 (sweep) | **unknown**; see section 4 | L |
| Lazy-zone half-width w | 0.10 | 0-0.3 | literature widths differ; width had little effect in a mandible remodeling study (Su et al. 2019) | L |
| c_form = c_res | 1.1e-4 (cort.), 7.7e-4 (canc.) /day | 2e-4-2e-3 (canc.) | turnover scaling (section 5) | L |
| rho bounds | cort. [0.5, 1.15], canc. [0.25, 2.0] | - | cortical porosity limits densification. Assumption | L |
| T, K | 365 d, 12 steps | - | one year, monthly updates | - |

## 7. What the model says

### Analytic threshold

At rho = 1, split each habit's stimulus into a cyclic part and a sustained
part: `Psi^m = A + n B`, where n = `static_equivalent_cycles`. Formation
starts where `Psi_h / Psi_ref = 1 + w`, which gives

    n_crit = [(1+w)^m A_ref - A_h] / [B_h - (1+w)^m B_ref]

(`craniofea.remodel.sensitivity.critical_static_cycles`). In an element whose
cyclic stimulus is dominated by swallowing and where the resting load has the
same spatial pattern as the swallowing load:

    n_crit ~ [(1+w)^m - 1] * n_swallow * (p_swallow / p_rest)^m
           = 0.46 * 600 * (15 kPa / 2 kPa)^4  ~  9e5 cycles/day

This estimate does not depend on geometry or stiffness. Chewing adds to A
and raises the threshold further. A 2 kPa posture has to be counted as about
a million load cycles per day before it moves any element out of the lazy
zone. That is roughly ten times more than holding the posture all day at
1 Hz.

### Results on the CT maxilla (Open-Full-Jaw patient 3)

Produced by `scripts/reproduce_results.py` with `configs/study.yaml`
(420,779 bone tets, JAX solver, orthotropic cortex, 12 monthly steps; every
linear solve converged to a relative residual below 1e-10).

| quantity | literature weighting (n = 1) | n = 1e7 (hypothetical) |
|---|---|---|
| palatal vault mean delta-rho after 12 months, mewing | 0 | +0.0065 |
| palatal vault mean delta-rho, mouth breathing | 0 | 0 |
| max \|S\|, mewing | 9.2e-8 | 0.47 |
| resting-posture share of the vault stimulus (volume average) | 5e-9 | 0.046 |
| reference habit max \|S\| (homeostasis check) | 1.4e-10 | - |

- At literature weighting, chewing supplies 98 % of the daily stimulus in the
  palatal vault and over 99.99 % elsewhere; swallowing supplies the rest of
  the vault's. At n = 1e7 the posture's share of the vault rises to 4.6 %.
- The analytic threshold on the CT mesh: in the palatal vault the first
  element responds at n = 5.3e5 and the median element at 7.0e6; the
  alveolar and basal regions need 1e6 or more for the first element and
  medians of 1e10 or more. The geometry-free estimate above (9e5) sits within
  this range.
- Mouth breathing never resorbs: its palatal mean delta-rho is exactly 0 at
  every weight tested (template sweep 0 to 1e7; CT at 1 and 1e7).

### Sensitivity on the parametric template

The template arch (15,426 tets) runs a full study in about a minute, which
makes sweeps practical. Sweeping `static_equivalent_cycles` from 0 to 1e7
gives a palatal contrast of exactly 0 up to 1e5, +6e-6 at 1e6 and +0.0066 at
1e7: the same threshold behaviour as the CT mesh.

<!-- MORRIS_RESULTS -->

### Interpretation

Under a Carter/Beaupré daily-stimulus model, a sustained posture can only
change maxillary density if (a) static loads are far more osteogenic than
the animal evidence suggests, or (b) the cycle exponent m is low (m = 2 brings
n_crit down to thousands). Mouth breathing as modelled here, with chewing and
swallowing unchanged, cannot cause resorption relative to a habit whose own
resting load is a few hundred pascal. If mouth breathing does affect
maxillary growth, this model points to mechanisms it does not include:
altered growth direction, sutural biology, soft-tissue pressure from the
lips and cheeks, airway and mandibular posture.

## 8. Limitations

- **Static loads.** The central parameter has no empirical value for human
  craniofacial bone. The model makes the dependence explicit. It does not
  resolve it.
- **Relative setpoint.** Because the setpoint is site-specific, the results
  are departures from the reference habit. They cannot say whether the
  reference habit itself is "adequate".
- **No growth.** Adolescent maxillary growth (sutural, appositional, drift),
  tooth eruption and changes of shape are not modelled. Only the density of a
  fixed geometry evolves.
- **No sutural biology.** The suture is a fixed soft layer. Expansion,
  fusion and the cellular response to tension are not modelled, although they
  are the most plausible routes by which tongue posture could act.
- **Maxilla only.** The mandible, cranial base, soft tissues and muscle
  attachments are absent. They enter only through the `fixed` boundary
  condition and the bite reaction.
- **Loading magnitudes.** Chewing force and frequency, the pressure of
  deliberate tongue posture, and the resting AP distribution are weakly
  constrained (see the confidence column). The adult values for swallowing
  frequency and pressure are assumed to apply to adolescents.
- **Material data.** Cortical and cancellous moduli come mainly from the
  mandible or from FE conventions, not from measurements of the adolescent
  maxilla.
- **Lazy-zone rate law.** The law is phenomenological. Its rate constants are
  scaled from turnover rather than fitted, so the time course is indicative
  only. The *sign* and the *threshold* of the response do not depend on the
  rates.

## References

- Beaupré GS, Orr TE, Carter DR (1990). An approach for time-dependent bone modeling and remodeling: theoretical development. *J Orthop Res* 8(5):651-661. PMID 2388105. Application: *J Orthop Res* 8(5):662-670, PMID 2388106.
- Campolongo F, Cariboni J, Saltelli A (2007). An effective screening design for sensitivity analysis of large models. *Environ Model Softw* 22(10):1509-1518. doi:10.1016/j.envsoft.2006.10.004
- Carter DR, Fyhrie DP, Whalen RT (1987). Trabecular bone density and loading history: regulation of connective tissue biology by mechanical energy. *J Biomech* 20(8):785-794. doi:10.1016/0021-9290(87)90058-3
- Carter DR, Hayes WC (1977). The compressive behavior of bone as a two-phase porous structure. *J Bone Joint Surg Am* 59(7):954-962.
- Fei T, Polacco RC, Hori SE, Molfenter SM, Peladeau-Pigeon M, Tsang C, Steele CM (2013). Age-related differences in tongue-palate pressures for strength and swallowing tasks. *Dysphagia* 28(4):575-581. doi:10.1007/s00455-013-9469-6
- Garijo N, et al. (2014). Numerical stability and convergence analysis of bone remodeling model. *Comput Methods Appl Mech Eng* 271:253-268. doi:10.1016/j.cma.2013.12.014
- Hayashi R, Tsuga K, Hosokawa R, Yoshida M, Sato Y, Akagawa Y (2002). A novel handy probe for tongue pressure measurement. *Int J Prosthodont* 15(4):385-388.
- Hill R (1952). The elastic behaviour of a crystalline aggregate. *Proc Phys Soc A* 65(5):349-354. doi:10.1088/0370-1298/65/5/307
- Hori K, Ono T, Tamine K, Kondo J, Hamanaka S, Maeda Y, Dong J, Hatsuda M (2009). Newly developed sensor sheet for measuring tongue pressure during swallowing. *J Prosthodont Res* (volume and pages not verified).
- Huiskes R, Weinans H, Grootenboer HJ, Dalstra M, Fudala B, Slooff TJ (1987). Adaptive bone-remodeling theory applied to prosthetic-design analysis. *J Biomech* 20(11-12):1135-1150. doi:10.1016/0021-9290(87)90030-3
- Lakatos É, Magyar L, Bojtár I (2014). Material properties of the mandibular trabecular bone. PMC4782746 (journal not verified).
- Lanyon LE, Rubin CT (1984). Static vs dynamic loads as an influence on bone remodelling. *J Biomech* 17(12):897-905.
- Lear CSC, Flanagan JB, Moorrees CFA (1965). The frequency of deglutition in man. *Arch Oral Biol* 10:83-99.
- Manolagas SC (2000). Birth and death of bone cells. *Endocr Rev* 21(2):115-137. PMID 10782361. Turnover table after Parfitt AM (1994) *J Cell Biochem* 55:273-286, as summarised in BioNumbers ID 109908.
- Morris MD (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics* 33(2):161-174.
- Po JMC, Kieser JA, Gallo LM, Tésenyi AJ, Herbison P, Farella M (2011). Time-frequency analysis of chewing activity in the natural environment. *J Dent Res* 90(10):1206-1210. doi:10.1177/0022034511416669
- Rice JC, Cowin SC, Bowman JA (1988). On the dependence of the elasticity and strength of cancellous bone on apparent density. *J Biomech* 21(2):155-168. doi:10.1016/0021-9290(88)90008-5
- Roldán SI, Restrepo LG, Isaza JF, Vélez LG, Buschang PH (2016). Are maximum bite forces of subjects 7 to 17 years of age related to malocclusion? *Angle Orthod* 86(3):456-461. doi:10.2319/051315-323.1
- Rubin CT, Lanyon LE (1984). Regulation of bone formation by applied dynamic loads. *J Bone Joint Surg Am* 66(3):397-402.
- Savoldi F, Xu B, Tsoi JKH, Paganelli C, Matinlinna JP (2018). Anatomical and mechanical properties of swine midpalatal suture in the premaxillary, maxillary, and palatine region. *Sci Rep* 8. doi:10.1038/s41598-018-25402-y
- Schwartz-Dabney CL, Dechow PC (2003). Variations in cortical material properties throughout the human dentate mandible. *Am J Phys Anthropol* (volume and pages not verified).
- Su et al. (2019). Numerical simulation of mandible bone remodeling under tooth loading: a parametric study. PMC6797806 (authors and journal details not verified).
- Thüer U, Sieber R, Ingervall B (1999). Cheek and tongue pressures in the molar areas and the atmospheric pressure in the palatal vault in young adults. *Eur J Orthod* 21(3):299-309. doi:10.1093/ejo/21.3.299
- Turner CH (1998). Three rules for bone adaptation to mechanical stimuli. *Bone* 23(5):399-407. doi:10.1016/S8756-3282(98)00118-5
- van Casteren A, Codd JR, Kupczik K, Plasqui G, Sellers WI, Henry AG (2022). The cost of chewing: the energetics and evolutionary significance of mastication in humans. *Sci Adv* 8(33):eabn8351. doi:10.1126/sciadv.abn8351
- Zeng J, Xu K, Gao X, Xu T. Pilot study of resting tongue pressure on a tongue anchorage pad (Peking University School of Stomatology; journal and year not verified).
