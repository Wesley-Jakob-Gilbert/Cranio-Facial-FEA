# Scientific & Mathematical Audit Report: Craniofacial FEA Pipeline

**Date:** March 30, 2026
**Focus:** Bone Remodeling Math and Mechanobiological Principles

## Executive Summary
An audit of the `cranio_FEA` pipeline reveals several structural inaccuracies in the mathematical implementation of bone remodeling. While the pipeline correctly establishes a modular framework for iterative FEA, the specific equations in `post/bone_remodeling.py` contain mathematical deviations from established mechanobiological theory (e.g., Huiskes' law) that will cause non-physical divergence, unbounded stiffness, and incorrect equilibrium states.

## 1. The Stimulus Function (Strain Energy Density per Mass)
**Current Implementation:**
```python
stimulus = (psi / psi_ref) - 1.0
```
Where `psi` is the Strain Energy Density (SED, energy per unit volume) extracted directly from the solver.

**Scientific Inaccuracy:**
In the standard Huiskes (1987) strain-adaptive bone remodeling theory, the mechanical stimulus that osteocytes sense is the **Strain Energy Density per unit mass** ($S = \frac{U}{\rho}$), not per unit volume.
If remodeling is driven by SED per unit volume ($U$) with a constant setpoint (`psi_ref`), the system lacks a stable equilibrium. As bone gets denser, its stiffness increases, which can *increase* $U$ under certain boundary conditions (stress-controlled), causing a positive feedback loop where dense bone gets infinitely denser. Dividing by $\rho$ creates the necessary negative feedback loop to reach homeostasis.

**Recommended Fix:**
Change the stimulus calculation to divide the apparent SED by the relative density:
```python
stimulus = ((psi / r0) / psi_ref) - 1.0
```

## 2. Unbounded Young's Modulus Scaling
**Current Implementation:**
```python
E_prev = max(1e3, E_base * (r0 ** n_pow))
```
With `r0` allowed to reach `rho_max = 1.5` and `n_pow_cortical = 3.0`.

**Scientific Inaccuracy:**
If $r_0$ reaches 1.5 in a cortical element where `E_base` is 15 GPa, the resulting Young's Modulus becomes $15 \text{ GPa} \times 1.5^3 = 50.625 \text{ GPa}$. 
Human cortical bone has a maximum physiological stiffness of ~20–25 GPa (longitudinal) and ~12–15 GPa (transverse). A stiffness of 50 GPa is physically impossible (closer to aluminum or glass than bone). The density multiplier `r0` must represent an apparent density fraction normalized such that `E` cannot exceed the biological maximum.

**Recommended Fix:**
Cap the maximum allowable modulus to a physiological limit, or redefine `E_base` to represent the *maximum* possible modulus when $r_0 = \rho_{max}$, rather than the baseline modulus at $r_0 = 1.0$.
```python
E_MAX_BONE_PA = 20.0e9 # 20 GPa physiological limit
E_new = min(E_MAX_BONE_PA, E_base * (r0 ** n_pow))
```

## 3. Lack of Explicit Time Integration (dt)
**Current Implementation:**
```python
drho = alpha_app * (stimulus - lazy_s)
```
The cycle updates density via `rho_new = r0 + drho`. 

**Scientific Inaccuracy:**
The remodeling rate `alpha_app` currently acts as a lumped parameter encompassing both the biological rate constant ($B$) and the time step size ($dt$). This makes the simulation cycles dimensionless and arbitrary. If the user changes `cycles: 12` to represent weeks instead of months, the `alpha` parameters must be manually recalibrated. 

**Recommended Fix:**
Introduce an explicit `dt` (time step duration) variable in `remodeling.yaml` (e.g., `dt_days: 30`) and update the rate equation:
```python
drho = rate_constant * dt * (stimulus - lazy_s)
```

## 4. Suture Remodeling Law (Yielding vs. Ossification)
**Current Implementation:**
```python
E_new = E_prev_suture * (1.0 - beta * (strain - strain_thresh))
```
Sutures *soften* (modulus decreases) when strain exceeds a threshold.

**Scientific Inaccuracy:**
While excessive strain can cause micro-tearing or fibrous yielding in sutures (which this formula captures), the primary developmental mechanism of sutures under mechanical stimulus is **osteogenesis (bone deposition) and eventual fusion/ossification**. Tension typically stimulates bone formation at the sutural margins, which would *increase* the effective modulus of the sutural zone over time, not decrease it. The current model simulates mechanical damage/yielding, not biological sutural growth.

**Recommended Fix:**
For a biological remodeling model, sutural modulus should increase (ossify) towards `E_cancellous` or `E_cortical` under appropriate tensile stimulus, and remain compliant or yield only if strains exceed a hyper-physiological damage threshold.

## 5. Fallback SED Approximation
**Current Implementation:**
```python
psi = (vm_e * vm_e) / (2.0 * E_prev)
```
Used when true SED (`ENER`) is unavailable.

**Scientific Inaccuracy:**
Approximating 3D Strain Energy Density using only the von Mises stress squared divided by $2E$ is strictly valid only for uniaxial stress states. In complex multi-axial states (like a maxilla under palate pressure), this formulation completely ignores the hydrostatic (volumetric) stress component, since von Mises stress is derived purely from the deviatoric stress tensor.

**Recommended Fix:**
Since the pipeline successfully requests and extracts true `ENER` from CalculiX, this fallback should be deprecated or replaced with a warning that it underestimates total SED in confined compression or triaxial tension.

---

## Action Plan for Immediate Codebase Fixes
1. **Fix the Stimulus Equation:** Update `post/bone_remodeling.py` to divide `psi` by `r0` before comparing to `psi_ref`.
2. **Cap Young's Modulus:** Enforce a physiological maximum of ~20-25 GPa in the `E_prev` and debug output calculations.
3. **Explicit Time Step:** Add a `dt` multiplier to the `drho` calculation to decouple biological rate from simulation step size.
