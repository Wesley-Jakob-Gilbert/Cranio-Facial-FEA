# Independent code and methods review

Reviewer: independent verification workstream; 2026-10-17. Initial package revision: `e1f8deebf2e8919365f78c503c7d244c48e7318b`, branch `validation-prep`; package, tests, scripts, configurations and reviewed methods documents remained unchanged while the coordinator advanced the branch to `0d53c02`. No package edits, commits or pushes were made by this reviewer; outputs are confined to `validation/review/` and this report. [Environment and diff check](../../validation/review/environment.txt)

## Summary verdict

**Verification: strong pass for the tested linear FE implementation. Validation: insufficient for a biological “mewing cannot work” conclusion.** The fast suite and full analytic verification reproduce the documented convergence and patch-test claims; the independent scalar calculation confirms the approximately \(9\times10^5\) swallowing-only threshold under the stated assumptions. [Test log](../../validation/review/pytest-fast.log), [verification results](../../validation/review/verification-full.json), [independent calculation](../../validation/review/independent_calibration.json)

The defensible headline is: **“Under this specified relative-strain, fourth-power daily-load model, with an assumed static weight of one and unchanged chewing/swallowing, posture does not cross the selected remodeling dead zone.”** Static weight one is a modeling convention, not a literature-calibrated human-maxillary dose; the original theory includes tissue-stress and surface-area terms not generally reproduced by this implementation. [METHODS §§4–5](../METHODS.md), [Turner](https://pubmed.ncbi.nlm.nih.gov/9823445/), [Beaupré](https://pubmed.ncbi.nlm.nih.gov/2388105/), [Stanford-law equations](https://link.springer.com/article/10.1007/s10237-022-01573-6)

**Do not label the CT threshold range independently reproduced:** no CT-scale study was run, no raw CT result arrays were present, and the repository itself reports failed remeshing and no CT convergence study. Moreover, the reported first-element threshold ignores the default smoothing operator. [Mesh limitations](../MESH.md), [threshold implementation, lines 59–110](../../src/craniofea/remodel/sensitivity.py), [environment](../../validation/review/environment.txt)

**New findings:** a reproducible setpoint-floor/smoothing defect; an incorrect Lear lower range (203, not 233 swallows/day); conflation of regional power-sum shares with stimulus shares; and an important population caveat—the dataset suppresses ages and lists Patient 3 as having **moderate periodontal disease**, so neither healthy-adolescent nor adult status is established. [Diagnostic](../../validation/review/review_diagnostics.json), [Lear](https://www.sciencedirect.com/science/article/abs/pii/0003996965900609), [share implementation](../../src/craniofea/remodel/stimulus.py), [Open-Full-Jaw paper, Table 2](https://cims.nyu.edu/gcl/papers/2022-OpenFullJaw.pdf)

## Test results

Python 3.12.13 was available and used in `validation/review/.venv`; editable installation with development dependencies succeeded, including JAX 0.11.2, NumPy 2.5.4, SciPy 1.18.1 and PyAMG 5.3.0. [Environment](../../validation/review/environment.txt), [installation log](../../validation/review/install-exact-command.log)

To obey the no-write-outside-validation rule, `src/`, `pyproject.toml` and `README.md` were copied unchanged into `validation/review/install-checkout/`, and `pip install -e ".[dev]"` was run there: this avoids setuptools creating `src/craniofea.egg-info` in the protected checkout. Tests use the original checkout via `tests/conftest.py`, and review execution wrappers prepend the original `src/`. [Install log](../../validation/review/install-exact-command.log), [conftest](../../tests/conftest.py), [verification wrapper](../../validation/review/run_verification.py), [diagnostic wrapper](../../validation/review/review_diagnostics.py)

```bash
# From repo root; constrain BLAS/OpenMP and keep generated files in validation/.
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/validation/review/mpl"
export TMPDIR="$PWD/validation/review/tmp"
validation/review/.venv/bin/python -m pytest -m "not slow" \
  -o cache_dir=validation/review/pytest-cache \
  --basetemp=validation/review/pytest-tmp \
  --junitxml=validation/review/pytest-fast.xml
validation/review/.venv/bin/python validation/review/run_verification.py
validation/review/.venv/bin/python validation/review/independent_calibration.py
validation/review/.venv/bin/python validation/review/review_diagnostics.py
```

| Run | Result | Evidence |
|---|---|---|
| Fast suite | **178 passed; 0 failed; 15 skipped; 2 deselected; 47.09 s** | [Full log](../../validation/review/pytest-fast.log) |
| Skips | CalculiX executable: 5; FEBio executable: 4; CT data: 3; network opt-in: 1; gmsh: 1; pyvista module at collection: 1 | [JUnit details](../../validation/review/pytest-fast.xml) |
| Full `scripts/verify_fem.py` | Completed all standard levels, both solver patch/reference/gradient checks; no CT benchmarks | [Log](../../validation/review/verification-full.log) |
| Independent scalar script | Energy/units, threshold versus numerical bisection, reference homeostasis and threshold edge-case assertions passed; **no `craniofea` imports** | [Script](../../validation/review/independent_calibration.py), [results](../../validation/review/independent_calibration.json) |
| Extra template check | At static weight 1: both habits’ vault mean density changes exactly zero; at \(10^7\): contrast 0.00656353 with 12 steps and 0.00654172 with 24 steps (about 0.33% relative difference) | [Diagnostic results](../../validation/review/review_diagnostics.json) |

There are no failing fast-test tracebacks to summarize; optional external-solver, visualization, remeshing and CT integration coverage remains unexecuted rather than passed. [JUnit details](../../validation/review/pytest-fast.xml)

## Verification checks

The original verification module was executed through a wrapper changing **only its output directory**, avoiding overwriting `docs/figures/`; the full, not `--quick`, level sequences were used. [Wrapper](../../validation/review/run_verification.py)

| Claim in VERIFICATION.md | Independent rerun | Verdict |
|---|---|---|
| Lamé rates 1.93 displacement / 0.93 stress | Same rounded rates; finest errors \(1.3774\times10^{-4}\) / \(1.6215\times10^{-2}\); 61,440 tets | Confirmed |
| Hanging-column rates 1.93 / 0.99 | Same rounded rates; finest errors \(4.4920\times10^{-5}\) / \(6.2419\times10^{-3}\) | Confirmed |
| Cantilever ratios 0.2194 → 0.9421 | Reproduced all five ratios; extrapolation order 1.47090 and limit ratio 1.00955 | Confirmed, but finest mesh still 5.79% below beam reference |
| Anisotropic jittered patch test | AMG max relative displacement/strain \(1.67\times10^{-15}\)/\(1.47\times10^{-14}\); JAX-CG \(7.15\times10^{-14}\)/\(1.36\times10^{-12}\) | Confirmed to rounding/platform variation |
| Direct-reference agreement / gradients | Largest reference displacement error \(7.91\times10^{-14}\), stress \(5.56\times10^{-14}\); gradient discrepancies \(1.08\times10^{-9}\) / \(2.35\times10^{-9}\) | Confirmed in substance |

All rerun values above are preserved in the [verification JSON](../../validation/review/verification-full.json) and [printed rates](../../validation/review/verification-full.log); comparison targets are in [VERIFICATION.md](../VERIFICATION.md).

These are **verification**, not anatomical validation: the two package solvers still share problem inputs and conventions, beam extrapolation is not a measured maxilla response, and low algebraic residual does not establish discretization accuracy on the CT anatomy. [Verification cases](../../scripts/verify_fem.py), [documented CT convergence gap](../MESH.md)

### Reproduced defect not caught by the fast suite

An element below the setpoint floor receives raw \(S=0\), but `run_study` then smooths it with neighboring bone and does not reapply the floor mask before updating density; in a six-tet diagnostic, its smoothed error becomes \(2/3\) and its density rate \(6.2333\times10^{-5}\)/day despite the documented promise that such elements remain fixed. This is a confirmed implementation/documentation inconsistency, **not evidence that it affected the reported CT headline**. [Diagnostic code/results](../../validation/review/review_diagnostics.py), [diagnostic JSON](../../validation/review/review_diagnostics.json), [law.py:37–64](../../src/craniofea/remodel/law.py), [simulate.py:235–240](../../src/craniofea/remodel/simulate.py)

## Re-derivation

### Units, daily aggregation and what was actually calibrated

For isotropic uniaxial stress, \(U=\sigma^2/(2E)\), so \(\sqrt{2U/E}=|\sigma|/E=|\epsilon|\): \(U\) has units J/m³ = Pa and \(E\) has units Pa. For orthotropy, replacing the directional modulus with the Hill modulus is an energy-equivalent scalar convention, **not generally the actual axial strain**. [Independent unit check](../../validation/review/independent_calibration.py), [implemented definition](../../src/craniofea/remodel/stimulus.py)

With \(n_i\) interpreted as dimensionless counts in a fixed one-day window,
\[
\Psi_\epsilon=\left(\sum_i n_i|\epsilon_i|^m\right)^{1/m}.
\]
If instead counts are treated dimensionally as rates, its units include day\(^{-1/m}\), not day\(^{-1}\); either consistent convention cancels in \(\Psi/\Psi^*\), so no missing seconds-to-days factor is required in that ratio. [Independent implementation](../../validation/review/independent_calibration.py)

The power aggregation is literature-supported, and \(m=4\) is a historical modeling choice; it is not an adolescent-maxilla measurement. [Carter 1987](https://pubmed.ncbi.nlm.nih.gov/3654678/), [NASA technical review of the formula and calibration](https://ntrs.nasa.gov/api/citations/20140012744/downloads/20140012744.pdf)

The Stanford/Beaupré form uses \(\bar\sigma=\sqrt{2EU}\), \(\Psi_\sigma=(\sum n_i\bar\sigma_i^m)^{1/m}\), tissue correction \(\Psi_t=(\hat\rho/\rho_{\rm abs})^2\Psi_\sigma\), and \(\dot\rho_{\rm abs}=\dot r\,S_v\hat\rho\), with surface apposition rate \(\dot r\), rather than a directly assigned relative-density rate. [Explicit published equations](https://link.springer.com/article/10.1007/s10237-022-01573-6), [original theoretical abstract](https://pubmed.ncbi.nlm.nih.gov/2388105/)

Consequently, for relative density \(r=\rho_{\rm abs}/\rho_0\) and \(E=E_0r^q\), this implementation gives normalized stimulus \((\Psi_\sigma/\Psi_{\sigma,0})r^{-q}\), versus \(r^{-2}\) for tissue stress; the factors coincide for \(q=2\), not cortical \(q=3\), and surface-area kinetics are still absent. At the initial density, however, each factor cancels between habits at the same element, so this distinction **does not by itself invalidate the initial threshold algebra**. [Algebra from the implemented modulus/stimulus laws](../METHODS.md), [Stanford-law equations](https://link.springer.com/article/10.1007/s10237-022-01573-6)

Setting \(\Psi_e^*=\Psi_{e,\mathrm{reference}}(r=1)\) makes the reference error exactly zero by construction; this is a baseline normalization, not independent identification of a biological attractor. Recomputing that reference for every swept static weight also means each sweep member represents a different assumed baseline equilibrium. [Independent homeostasis check](../../validation/review/independent_calibration.json), [simulate.py:187–191](../../src/craniofea/remodel/simulate.py), [sensitivity.py:44–53](../../src/craniofea/remodel/sensitivity.py)

The turnover arithmetic is correct: \(0.04/365=1.09589\times10^{-4}\)/day and \(0.28/365=7.67123\times10^{-4}\)/day; identifying turnover with net density gain at one arbitrarily chosen excess stimulus, and equating formation and resorption slopes, is **not** a calibration supplied by those data. [Independent arithmetic](../../validation/review/independent_calibration.json), [BioNumbers/Manolagas/Parfitt](https://bionumbers.hms.harvard.edu/bionumber.aspx?id=109908)

### Sustained-load threshold

Independently define \(A=\sum_{\rm cyclic}n_i\epsilon_i^m\), \(B=\sum_{\rm sustained}\epsilon_i^m\), and \(k=(1+w)^m\); comparing \(A_h+nB_h=k(A_r+nB_r)\) gives
\[
n_{\rm crit}=\frac{kA_r-A_h}{B_h-kB_r}.
\]
This is the onset boundary (rate is still zero at equality): if the numerator is negative, formation already occurs at zero static weight; for nonnegative numerator and nonpositive denominator, there is no finite formation crossing, assuming a nonzero reference stimulus. [Independent implementation and bisection checks](../../validation/review/independent_calibration.py)

With identical swallowing/resting load shapes, unchanged cyclic loads, negligible chewing/gravity, \(p_s=15\) kPa, \(p_h=2\) kPa, \(p_r=0.2\) kPa, \(n_s=600\), \(m=4\), \(w=0.10\):
\[
n_{\rm crit}
=\frac{(1.1^4-1)\,600\,(15/2)^4}
{1-1.1^4(0.2/2)^4}
=881{,}193.86\ \text{equivalent cycles/day}.
\]
Neglecting the reference pressure gives 881,064.84, agreeing with the rounded \(9\times10^5\) claim; direct bisection agrees with the exact expression to floating-point accuracy. [Independent results](../../validation/review/independent_calibration.json)

Useful counterchecks: \(m=2\) gives 7,174/day, \(m=8\) gives \(6.869\times10^9\)/day, \(w=0\) makes the onset boundary zero, and 86,400/day gives \(S=0.01119<0.10\) in this scalar example; the \(m=4\) threshold is 10.20 times that illustrative 1-Hz-day count. [Independent results](../../validation/review/independent_calibration.json)

**The quoted CT interval \(5.3\times10^5\)–\(7.0\times10^6\) does not follow from pressures alone.** It is an unverified-here geometry-dependent minimum/median pair, not a confidence interval or a universal physiological threshold; the actual rest/swallow AP shapes also differ (2 versus 1). Chewing raises the scalar threshold, but importing the regional “98%” share into a typical element is invalid: if a particular element really had chewing equal to 49 times its swallowing term, this scalar threshold would be \(4.406\times10^7\), not \(7\times10^6\). [Reported CT results/configuration](../METHODS.md), [independent example](../../validation/review/independent_calibration.json)

Default smoothing further invalidates calling the raw element threshold the actual onset map: an equal-weight two-element illustrative average moves the boundary from 0.4641 to 1.0736 (a factor 2.31). A smoothed threshold requires solving \([H(\Psi_h(n)/\Psi_r(n)-1)]_e=w\), with floor masking handled consistently. [Independent example](../../validation/review/independent_calibration.json), [acknowledged omission in sensitivity.py:72–76](../../src/craniofea/remodel/sensitivity.py)

Two interpretation corrections:

- “98% of daily stimulus” is actually a share of \(\sum_eV_e\Psi_e^m\), not a volume-average of local fractional shares or an additive share of \(\Psi\); for one element with that power share, removing the other 2% changes \(\Psi\) by only 0.504%, whereas removing chewing leaves 37.6% of \(\Psi\). [Share code](../../src/craniofea/remodel/stimulus.py), [independent arithmetic](../../validation/review/independent_calibration.json)
- “Mouth breathing never resorbs” must be bounded to the tested scenarios: in the same no-gravity scalar example it crosses the resorption boundary at \(9.95\times10^9\) static-equivalent cycles/day, although not anywhere near \(10^7\); this is a mathematical counterexample, not a plausible exposure. [Independent result](../../validation/review/independent_calibration.json)

## Parameter audit table

All 25 rows in METHODS §6 are covered below. **Y** means the numerical value/law or explicitly identified convention is supported; it does not validate transfer to adolescents. **N** means the exact default lacks source calibration, is an assumption, or disagrees with the source; source existence and supported ranges are distinguished in the notes. “None” refers to the absence of a cited measurement in the original table. [Parameter table under review](../METHODS.md)

| Parameter | Value | Cited source | Verified Y/N | Note | URL |
|---|---|---|---|---|---|
| Cortical isotropic E | 15 GPa; 10–20 | “Typical FE value,” no named citation | N | Missing original source; supplemental review supports 13.7–17 GPa as common **mandibular FE assignments**, not adolescent-maxilla measurement. | [Mandible-model review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7818111/) |
| Orthotropic E1/E2/E3 | 20/13/11 GPa | Schwartz-Dabney & Dechow 2003 | N | Exists, 120(3):252–277; adult dentate **mandible**, longitudinal E3 20–30 GPa, 50–100% stiffer. Exact triplet/axes not verified; model relabels strongest direction E1 and maps it to maxillary AP. | [Primary abstract](https://pubmed.ncbi.nlm.nih.gov/12567378/) |
| G12/G13/G23 | 5/4/3.5 GPa | None; representative | N | UNVERIFIED measured values; must remain assumptions, including their coupling with the chosen orthotropic E and Poisson ratios. | [Original assumption](../METHODS.md) |
| Bone Poisson ratios | 0.30 | None; conventional | Y (convention only) | Supplemental review confirms common 0.3 use, not patient-specific measurement; measured mandibular directional values vary. | [Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7818111/) |
| Cancellous E | 1.0 GPa; 0.3–2 GPa | Lakatos et al. 2014 | N | Exists: *J Med Eng* 2014:470539. Measured apparent moduli 6.9–199.5 MPa, fresh middle-aged male mandibular specimens; default 1000 MPa is ~5× their maximum, and **entire sweep is above this study’s range**. Other directional measurements differ; do not confuse tissue nanoindentation modulus with apparent continuum modulus. | [Full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC4782746/) |
| Density exponents | 3 cortical / 2 cancellous | Carter & Hayes 1977; Rice et al. 1988 | Y (laws only) | Both exist; cubic modulus relation was obtained from human/bovine trabecular compression and generalized, while Rice’s pooled human/bovine cancellous analysis gives square dependence. Neither establishes adolescent-maxillary cortical exponent 3. Specific hip/spine provenance is not established by these abstracts. | [Carter & Hayes](https://pubmed.ncbi.nlm.nih.gov/561786/); [Rice](https://pubmed.ncbi.nlm.nih.gov/3350829/) |
| Suture E | 10 MPa; 1–100 | Savoldi et al. 2018 | N | Exists: *Sci Rep* 8:7073; 12.5/31.3/98.5 MPa reproduced, but **9–12-month swine composite bone–suture–bone bending**, not isolated adolescent human suture tissue. The stated FE 0.7–69 MPa range was not verified from this source. | [Primary article](https://www.nature.com/articles/s41598-018-25402-y) |
| Suture nu | 0.45 | None; assumption | N | No tissue-specific source; “near-incompressible” is a constitutive choice, not a measured value. | [Original assumption](../METHODS.md) |
| Mass density cortical/cancellous/suture | 1850/1000/1100 kg/m³ | None; representative | N | 1850 kg/m³ is 1.85 g/cm³, corroborated only indirectly by mandibular cortical literature; exact remaining values UNVERIFIED. Downgrade M confidence for the unsourced triplet. | [Secondary density report](https://pmc.ncbi.nlm.nih.gov/articles/PMC2743800/) |
| Reference resting pressure | 200 Pa; 50–500 | Zeng et al.; Thüer et al. 1999 | N (range supported) | Zeng located: *J Oral Rehabil* 2015;42:414–419, ages 22–33, upright 113.07–183.94 Pa; supine up to 201.69 Pa on an appliance. Thüer: median 1 gf/cm²≈98.07 Pa **at teeth**, not vault contact. Neither calibrates uniform maxillary reference 200 Pa. | [Zeng](https://pubmed.ncbi.nlm.nih.gov/25683451/); [Thüer](https://academic.oup.com/ejo/article/21/3/299/480755) |
| Deliberate resting pressure | 2 kPa; 1–4 | Hayashi et al. 2002 | N | Exists, but measures maximum voluntary pressure 10–41 kPa in adults 24–84, not sustained posture. 2 kPa is **4.9–20%** of that range, not generally 5%. | [Hayashi](https://pubmed.ncbi.nlm.nih.gov/12170854/) |
| Resting AP ratio | 2; 0.5–3 | Zeng; unnamed conflicting study | N | Zeng supports upright anterior>posterior qualitatively; exact ratio 2 not calibrated. Conflicting study identified: Yu & Gao 2019, 46:249–256, front-to-back increase. | [Zeng](https://pubmed.ncbi.nlm.nih.gov/25683451/); [Yu & Gao](https://onlinelibrary.wiley.com/doi/full/10.1111/joor.12741) |
| Swallow pressure | 15 kPa; 5–30 | Hayashi; Fei 2013; unnamed secondary studies | N (range supported) | Hayashi’s 3–27 kPa water-swallow range confirmed; 15 is an adopted value within it, not a whole-palate simultaneous pressure map. Fei confirms saliva>water; sensor-sheet 10–16 kPa claim remains UNVERIFIED here. | [Hayashi](https://pubmed.ncbi.nlm.nih.gov/12170854/); [Fei](https://pubmed.ncbi.nlm.nih.gov/23677389/) |
| Swallow AP ratio | 1 | Hori et al. 2009 | N | Exists: *J Prosthodont Res* 53(1):28–32. Abstract supports corrected sheet-vs-sensor agreement, **not equality across palatal channels**; equal pressure/anterior timing require full-text evidence. | [Hori](https://pubmed.ncbi.nlm.nih.gov/19318068/) |
| Swallows/day | 600; 230–1000 | Lear et al. 1965 | Y (rounded mean) | Primary abstract: 20 subjects, mean 585/day, **203–1008**, not 233–1008; 600 is reasonable rounding, but the range citation needs correction and age applicability needs confirmation. | [Primary abstract](https://www.sciencedirect.com/science/article/abs/pii/0003996965900609) |
| Chewing bite reaction | 100 N; 50–250 | Roldán et al. 2016 | N | Exists and age-16 peak verified, but measures maximum voluntary bite, not chewing: molar peaks 62.6/57.7 kg-force (~614/566 N). Does not establish 100 N per chewing cycle or its spatial distribution. | [Primary article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8601729/) |
| Chews/day | 1500; 200–3300 | Po 2011; van Casteren 2022 | N | 1.57 Hz confirmed; 35.3 min/day (7.2–75.7) is prior Organ et al. data quoted by van Casteren. 1500 corresponds to 15.92 min; 35.3 min gives ~3325 cycles, so default is not the cited mean. | [Po](https://pubmed.ncbi.nlm.nih.gov/21810620/); [van Casteren](https://pmc.ncbi.nlm.nih.gov/articles/PMC9385136/); [arithmetic](../../validation/review/independent_calibration.json) |
| Molar weighting | 3 | None; assumption | N | No calibrated posterior/anterior force ratio supplied. | [Original assumption](../METHODS.md) |
| Gravity | 9.81 m/s², −si | None in table | Y (rounded magnitude) | Standard acceleration is 9.80665 m/s². Constant −si assumes a head orientation and does not describe an all-day posture history. | [NIST](https://physics.nist.gov/cuu/Constants/Table/allascii.txt); [loading definition](../METHODS.md) |
| Stimulus exponent | 4; 2–8 | Carter 1987; Beaupré 1990 | Y (historical convention) | Sources exist; NASA review explicitly attributes m=4 to Beaupré’s interpretation of earlier animal data. Not a fitted craniofacial exponent; proximal-femur modeling provenance. | [Carter](https://pubmed.ncbi.nlm.nih.gov/3654678/); [Beaupré](https://pubmed.ncbi.nlm.nih.gov/2388105/); [formula review](https://ntrs.nasa.gov/api/citations/20140012744/downloads/20140012744.pdf) |
| Static-equivalent cycles | 1/day; 0–10⁷ | Rubin/Lanyon; Lanyon/Rubin; Turner | N | Sources exist and support dynamic/static differences, **not a conversion to one cycle/day**; avian long-bone evidence is not a human maxillary calibration. | [1984 static experiment](https://pubmed.ncbi.nlm.nih.gov/6520138/); [1984 dynamic experiment](https://pubmed.ncbi.nlm.nih.gov/6699056/); [Turner](https://pubmed.ncbi.nlm.nih.gov/9823445/) |
| Lazy-zone half-width | 0.10; 0–0.3 | Su et al. 2019 | N | Exists: Su, Yuan, Yang, Du, *Sci Rep* 9:14887. Default **15%**, tested 5–45%; **2D mandible SED/mass**, not this fourth-root stimulus. Reduced effect on mean equilibrium density does not establish threshold insensitivity here. | [Su](https://pmc.ncbi.nlm.nih.gov/articles/PMC6797806/) |
| Formation=resorption slopes | 1.1×10⁻⁴ / 7.7×10⁻⁴ day⁻¹ | Parfitt 1994 via Manolagas 2000/BioNumbers | N (arithmetic Y) | 4%/28% annual turnover verified; division by 365 correct. Transfer from adult whole-skeleton turnover to net adaptive maxillary density rates and symmetric slopes is an assumption. | [BioNumbers with full source chain](https://bionumbers.hms.harvard.edu/bionumber.aspx?id=109908) |
| Relative-density bounds | cortical [0.5,1.15]; cancellous [0.25,2] | None; assumption | N | Not validated against measured porosity/mineral density; relative initial density is not a physical maximum-density calibration. | [Original assumption](../METHODS.md) |
| Duration / steps | 365 days / 12 | None; numerical design | N (biological calibration) | A selected horizon and Euler step, not a sourced biological parameter; 24-step template check changes high-weight contrast only ~0.33%, but does not establish CT/all-parameter convergence. | [Configuration](../../configs/study.yaml); [diagnostic](../../validation/review/review_diagnostics.json) |

**Transfer audit:** documented transfers are primarily adult mandible→maxilla, swine composite suture→human suture layer, whole-skeleton adult turnover→adaptive density kinetics, and a proximal-femur/long-bone daily-stimulus framework→craniofacial modeling. This review did **not** find evidence that the exact defaults are directly measured hip/spine values; do not invent that provenance, and do not substitute tissue-scale modulus measurements for apparent cancellous modulus. [Mandible data](https://pubmed.ncbi.nlm.nih.gov/12567378/), [suture data](https://www.nature.com/articles/s41598-018-25402-y), [turnover](https://bionumbers.hms.harvard.edu/bionumber.aspx?id=109908), [formula provenance](https://ntrs.nasa.gov/api/citations/20140012744/downloads/20140012744.pdf), [apparent-modulus study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4782746/)

## Threats ranked by severity with falsification tests

These are proposed tests, not completed biological validation.

| Rank | Threat and evidence | Concrete test / falsification criterion |
|---|---|---|
| 1 — Critical | **Static-dose equivalence and unlimited power-law cycle accumulation.** The supporting animal experiment showed no additional response from 36 to 1800 cycles; the implemented formula still accumulates indefinitely. [Rubin & Lanyon](https://pubmed.ncbi.nlm.nih.gov/6699056/), [stimulus code](../../src/craniofea/remodel/stimulus.py) | Pre-register power-law, saturating-cycle, rest-recovery and static-zero alternatives, first fitting independent animal dose-response observations. Recompute the threshold under every admissible model. A materially different ordering/onset invalidates a model-independent million-cycle headline. |
| 2 — Critical | **Endpoint mismatch: density is not adolescent facial growth.** Geometry is fixed and suture biology is disabled. [METHODS limitations](../METHODS.md) | Limit context of use to density, or add a growth/suture model tested against independent longitudinal arch-width, palatal-shape and maturation observations. Measured shape change without density change directly falsifies inference from this null density result to “no facial effect.” |
| 3 — High | **Unmeasured actual loading and spatial patterns.** Maximum bite/tongue tests do not measure the assumed sustained posture or ordinary chewing; all habits keep other loads unchanged. [Roldán](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8601729/), [Hayashi](https://pubmed.ncbi.nlm.nih.gov/12170854/), [configuration](../../configs/study.yaml) | Use calibrated pressure sensors and noninvasive chewing-event recordings in an ethically approved pilot; estimate distributions, contact areas and temporal covariance rather than single peaks. Rerun full measured histories, including unilateral bite and altered swallow/chew patterns. If the measured uncertainty interval crosses a response boundary, the robust-null headline fails. |
| 4 — High | **Assumed homeostasis, dead-zone width and exponent.** Reference equilibrium is imposed; width zero and m=2 dramatically alter onset. [Setpoint code](../../src/craniofea/remodel/simulate.py), [independent counterchecks](../../validation/review/independent_calibration.json) | Lock a reference inferred from independent baseline/longitudinal data; compare fixed versus recalibrated setpoints, widths 0/0.05/0.10/0.20 and m=2/4/8, reporting threshold surfaces rather than only Morris means. A non-null result at an independently supported setting falsifies the claimed insensitivity. |
| 5 — High | **CT mesh convergence absent; tet4 bending remains stiff.** Remeshing failed; the finest cantilever is still 5.79% below its beam reference. [MESH](../MESH.md), [rerun](../../validation/review/verification-full.json) | Produce three CT refinements or a quadratic-element cross-check, preserving tissue thickness/contact labels; compare vault stimulus ratios and weighted threshold quantiles, not residual alone. Predefine <5% change on two finest levels; failure blocks quantitative CT thresholds. |
| 6 — High | **Cut-face supports and load transfer may control the result.** Cropped maxilla is clamped, teeth/PDL removed, and bite reaction distributed over sockets/crest. [MESH surface definitions](../MESH.md) | Compare clamped versus compliant buttress springs versus extended cranial geometry, and retained PDL/teeth versus bone-only loading; check force/moment balance and measured strain patterns. A changed sign or >2× median threshold across plausible support models invalidates the current quantitative claim. |
| 7 — High | **Single patient, unknown age, periodontal disease.** Patient 3 has moderate periodontal disease; age was removed, not verified adult or adolescent. [Dataset paper](https://cims.nyu.edu/gcl/papers/2022-OpenFullJaw.pdf) | Obtain authorized age-band/maturation and disease metadata; replicate on at least three independent suitable jaws initially, then quantify cohort variability. A suitable adolescent case responding under the predeclared exposure range falsifies generalization from Patient 3; no admissible age data means remove adolescent-specific claims. |
| 8 — High | **Material assignments and scale transfer.** Cancellous sweep misses the cited apparent-modulus range; suture modulus comes from a composite swine bending specimen. [Lakatos](https://pmc.ncbi.nlm.nih.gov/articles/PMC4782746/), [Savoldi](https://www.nature.com/articles/s41598-018-25402-y) | Extend apparent cancellous modulus to 7–200 MPa; independently vary cortical orientation/thickness and suture width/tissue modulus with correlated constraints. Validate displacements/strains against a bench jaw before remodeling. A fit requiring parameters outside the reported ranges or a reversed load dominance refutes current calibration. |
| 9 — Medium/high | **Constitutive adaptation and time scale are not validated Beaupré kinetics.** Current-strain normalization and direct density slopes replace tissue-stress/specific-surface terms; turnover is not net gain. [Original theory](https://pubmed.ncbi.nlm.nih.gov/2388105/), [explicit equations](https://link.springer.com/article/10.1007/s10237-022-01573-6), [turnover source](https://bionumbers.hms.harvard.edu/bionumber.aspx?id=109908) | Implement the original tissue-stress/surface-area law alongside this strain variant; fit both to held-out load-change data and test 12/24/48/96 steps. Divergent predictions outside measurement uncertainty invalidate claims of calibrated one-year effect sizes, though not the shared initial algebra. |
| 10 — Medium/high | **Raw thresholds, smoothing and inactive-element handling disagree.** Threshold map ignores smoothing; the floor bug is reproduced. [Threshold source](../../src/craniofea/remodel/sensitivity.py), [diagnostic](../../validation/review/review_diagnostics.json) | Add regression tests preserving inactive densities; solve smoothed first-step thresholds and verify with ±1% exposure perturbations. Repeat with mesh-independent physical smoothing radius. If minimum/median thresholds shift beyond the predefined tolerance, withdraw the raw map as onset of the production law. |

## Issues for Claude Code

- **P1:** Preserve the setpoint-valid mask through smoothing and density updates; add a neighboring-high-stimulus regression test (`src/craniofea/remodel/law.py:37–64`, `simulate.py:235–240`; reproduced in [review_diagnostics.py](../../validation/review/review_diagnostics.py)).
- **P1:** Label `critical_static_cycles` output as unsmoothed onset or implement smoothed/floor-aware numerical roots; expose volume-weighted threshold quantiles and ±1% onset tests (`src/craniofea/remodel/sensitivity.py:59–110`, `docs/METHODS.md:262–266`; [current omission](../../src/craniofea/remodel/sensitivity.py)).
- **P1:** Replace “literature weighting/calibration” for static weight 1 with “assumed convention,” and limit the conclusion to fixed-geometry relative density (`docs/METHODS.md:125–137,245–268,301–309`; [Turner](https://pubmed.ncbi.nlm.nih.gov/9823445/)).
- **P1:** Name and document the strain-based adaptation as a variant, including differences from tissue-stress and surface-area kinetics (`src/craniofea/remodel/stimulus.py:3–27`, `docs/METHODS.md:103–120,156–182`; [Stanford equations](https://link.springer.com/article/10.1007/s10237-022-01573-6)).
- **P1:** Put CT mesh-convergence status, unknown patient age and moderate periodontal disease beside the CT headline, not only in limitations (`docs/METHODS.md:245–268`; [dataset Table 2](https://cims.nyu.edu/gcl/papers/2022-OpenFullJaw.pdf), [MESH.md:278–294,356–358](../MESH.md)).
- **P2:** Rename output/figure language to “fraction of volume-integrated fourth-power stimulus,” not “volume average” or additive daily stimulus share (`src/craniofea/remodel/stimulus.py:74–82`, `docs/METHODS.md:256,259–261`; [implementation](../../src/craniofea/remodel/stimulus.py)).
- **P2:** Correct Lear range 233→203, add verified bibliography for Zeng/Hori/Lakatos/Schwartz-Dabney/Su, and change “2 kPa ≈5%” to “4.9–20% of cited maximum range” (`docs/METHODS.md:205–210,346–367`; [Lear](https://www.sciencedirect.com/science/article/abs/pii/0003996965900609), [Hayashi](https://pubmed.ncbi.nlm.nih.gov/12170854/)).
- **P2:** Expand cancellous sensitivity below 0.3 GPa and label suture values as composite-specimen measurements rather than tissue calibration (`configs/study.yaml:109–110`, `docs/METHODS.md:200–202`; [Lakatos](https://pmc.ncbi.nlm.nih.gov/articles/PMC4782746/), [Savoldi](https://www.nature.com/articles/s41598-018-25402-y)).
- **P2:** Qualify the effective-strain identity as the **magnitude** of axial strain for **isotropic** uniaxial stress (`src/craniofea/remodel/stimulus.py:10`, `docs/METHODS.md:109`; [definition](../../src/craniofea/remodel/stimulus.py)).
- **P2:** Change pressure-normalization claim from “resultant force = mean pressure × area” to “integrated scalar pressure = mean pressure × area”; curved-face vector resultant is \(-\sum_f p_fA_f\mathbf n_f\) and need not remain invariant (`docs/METHODS.md:83–90`; [defined pressure law](../METHODS.md)).
- **P2:** Bound “mouth breathing never/cannot resorbs” wording to the tested weight range and unchanged other habits (`docs/METHODS.md:267–268,304–306`; [independent counterexample](../../validation/review/independent_calibration.json)).
- **P2:** Add a `--output-dir` option to verification and lock a reproducibility environment; current script writes protected figures by default (`scripts/verify_fem.py:29,main`; [review wrapper](../../validation/review/run_verification.py)).

## Needs Wes

- Decide whether the paper’s context of use is **model verification and conditional density hypotheses**, or biological claims about adolescent growth; the current implementation supports only the former without additional validation. [METHODS limitations](../METHODS.md)
- Ask the dataset authors whether authorized age-band/suture-maturation information is obtainable and whether Patient 3’s moderate periodontal disease makes it unsuitable for the intended healthy-adolescent comparison; do not assume adulthood from missing metadata. [Dataset paper](https://cims.nyu.edu/gcl/papers/2022-OpenFullJaw.pdf)
- Obtain clinician input, an appropriate ethics determination and calibrated equipment before gathering human pressure/history or longitudinal validation data; do not recommend deliberate loading or new radiation exposure as a DIY validation experiment.
- Personally resolve commercial-use/redistribution permission before MorphoGenix uses CT-derived assets: the project reports CC BY-NC-SA 4.0, but this reviewer did not independently inspect the underlying license text. **License terms independently UNVERIFIED in this workstream.** [Project licensing statement](../MESH.md), [upstream repository](https://github.com/diku-dk/Open-Full-Jaw)
- Approve access to enough compute/meshing expertise for CT convergence and an independent solver comparison; those are still missing, whereas the modest template time-step check is encouraging. [CT gap](../MESH.md), [template check](../../validation/review/review_diagnostics.json)
- Preserve the small logs/JSON/scripts as evidence, but **do not stage the entire `validation/review/` directory**: it also contains the local venv, dependency cache and editable-install mirror; no cleanup was performed.
