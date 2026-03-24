# Scientific Caveats and Limitations
# cranio_FEA — Adolescent Craniofacial Remodeling Model

> **Read before interpreting any output from this pipeline.**
> This model is a mechanobiological hypothesis explorer, not a clinical predictor.
> All outputs are directional and comparative — not quantitative predictions for
> any specific individual.

---

## 1. Geometry Is a Toy Approximation

The maxillary geometry is a parameterically warped hexahedral block with
analytic anatomical features (anterior taper, palatal groove, alveolar ridge lift,
posterior skew). It is **not** derived from CT or CBCT imaging. It does not
replicate the true geometry of any patient's maxilla, including:

- The three-dimensional arch form
- Cortical shell thickness variation
- Trabecular architecture
- Dental roots and alveolar bone detail
- Zygomatic process and adjacent cranial structures

**Implication:** Stress magnitudes and displacement values are not anatomically
meaningful in absolute terms. Only relative comparisons between load cases
(mewing vs. mouth-breathing) carry directional scientific meaning.

---

## 2. Suture Biology Is Modeled as Mechanical Compliance Only

The midpalatal and lateral sutural zones are modeled as thin compliant elastic
layers (lower Young's modulus than bulk bone). This captures the mechanical
effect of suture presence (stress concentration, differential compliance) but
does not model:

- Osteoblast/osteoclast cell activity
- Growth factor signaling (TGF-β, FGF, Wnt pathways)
- Vascular supply or tissue fluid flow
- Suture morphology (interdigitation, oblique orientation)
- Progressive suture fusion with age (ossification timeline)

**Implication:** The model cannot distinguish between a suture that is
mechanically compliant because it is open (adolescent) versus one that
would be biologically responsive to mechanical stimulus. The sutural
compliance parameter `E_suture_pa` in `configs/material.yaml` is a
coarse proxy for age — not a validated biological parameter.

---

## 3. Tongue Force Magnitude and Duration Are Estimated

Resting tongue pressure against the palate is not directly measured in this
model. The `tongue_kpa` parameter (2.0 kPa for the mewing scenario) is set
within the range reported in intraoral pressure measurement studies
(~0.1–4 kPa), but:

- Real tongue pressure is episodic and cyclic, not constant
- The model applies a static sustained load — equivalent to 24/7 uninterrupted
  pressure, which is not physiologically accurate
- Individual variation in tongue strength, posture, and contact area is large
- The exact pressure required to induce sutural bone response is unknown

**Implication:** The absolute magnitude of predicted displacements and stresses
should not be used to estimate "how much force mewing produces." The model
can only say: under these assumed loading conditions, the mechanical stimulus
at the suture is X% higher than the reference stimulus.

---

## 4. The Remodeling Law Is Established; Its Parameter Values Are Not

The Huiskes strain-adaptive remodeling law (SED-stimulus driven density update)
is a well-validated framework in computational biomechanics. It has been applied
to hip prostheses, vertebral bodies, and long bones. However, the parameter
values used here (`psi_ref_pa`, `alpha`, `rho_min`, `rho_max`, `n_power`) are:

- Sourced from the trabecular bone remodeling literature (primarily hip/spine)
- **Not validated** for the craniofacial skeleton
- **Not validated** for adolescent suture-adjacent bone
- Set to plausible defaults, not to patient-specific measurements

**Implication:** The density trajectories shown in the animation represent what
the Huiskes law *predicts given these parameters* — not what has been measured
in craniofacial bone. Different reasonable parameter choices would give
quantitatively different outputs (though directional trends should be robust).

---

## 5. The Mouth-Breathing Scenario Is a Zero-Load Approximation

The mouth-breathing case is modeled as `tongue_kpa: 0.0` — zero palate pressure.
This is a simplification. Real mouth-breathing does not produce zero tongue
force; it produces a different tongue posture (tongue resting low in the oral
cavity) which may produce:

- Downward forces on the mandible rather than upward forces on the palate
- Altered buccal muscle loading patterns
- Different airway mechanics that indirectly affect bone loading

The comparison in this model is therefore between:
- **Mewing:** sustained 2 kPa upward palate load
- **Mouth-breathing proxy:** zero palate load (no tongue-palate contact)

This is the *mechanically correct* analog for absent tongue-palate contact, but
it is not a full model of the mouth-breathing craniofacial loading environment.

---

## 6. Material Is Homogeneous Isotropic Elastic

Bone is modeled as a single isotropic linear elastic material (E = 1 GPa,
ν = 0.3). Real bone is:

- **Anisotropic:** cortical bone has directional stiffness
- **Heterogeneous:** cortical shell (~10–20 GPa) vs. trabecular core (~0.1–3 GPa)
- **Viscoelastic:** exhibits creep and stress relaxation
- **Nonlinearly elastic** at higher strains

The current model has no cortical/trabecular split. The 1 GPa value is a
bulk average approximation. Peak stresses at boundaries and corners are
likely underestimated; trabecular zone stresses may be overestimated.

---

## 7. Confounders Not Modeled

The following factors influence craniofacial bone development in adolescents
and are entirely absent from this model:

| Factor | Effect | Notes |
|--------|--------|-------|
| Genetics | Dominant driver of facial morphology | Not parameterizable |
| Hormonal milieu | Growth hormone, sex steroids affect osteoblast activity | Not modeled |
| Nutrition | Calcium, vitamin D affect bone mineralization | Not modeled |
| Dental occlusion | Masticatory forces are the dominant mechanical input | Jaw muscle proxy only |
| Sleep posture | Hours of positional loading on the face | Not modeled |
| Lip competence | Buccal pressure from lip seal affects transverse width | Not modeled |
| Nasal airway | Nasal breathing generates negative intraoral pressure | Not modeled |

---

## 8. No Clinical Validation

This model has not been validated against:

- Longitudinal CBCT studies of palate morphology change
- Rapid maxillary expansion treatment outcomes
- Orthotropic treatment case series
- Any individual patient data

Without validation, the model cannot make quantitative predictions. It can
only demonstrate the *direction* and *relative magnitude* of mechanical
effects under specified assumptions.

---

## Appropriate Uses of This Model

| Use | Appropriate? |
|-----|-------------|
| Visualizing the mechanical hypothesis for tongue posture | Yes |
| Comparing relative stress distributions between loading scenarios | Yes |
| Sensitivity analysis to understand which parameters matter most | Yes |
| Teaching tool for craniofacial biomechanics | Yes |
| Hypothesis generation for future clinical research | Yes |
| Predicting clinical outcomes for a specific patient | **No** |
| Supporting marketing claims about any device or technique | **No** |
| Replacing orthodontic or medical evaluation | **No** |

---

*Generated by the cranio_FEA pipeline. For parameter details see
`configs/material.yaml`, `configs/geometry.json`, and `configs/remodeling.yaml`.*
