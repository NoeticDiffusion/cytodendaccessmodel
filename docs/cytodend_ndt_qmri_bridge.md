# Cytodend Access Model — Compact Mathematical Reference (LLM target)

> Purpose: dense LLM-readable reference for the branch-resolved cytoskeletal-dendritic
> accessibility model (cytodend_accessmodel repo) and its formal coupling to the Noetic
> Diffusion Theory (NDT) layer stack and quantitative MRI (qMRI).  
> Version: June 2026. Source repo: NoeticDiffusion/cytodendaccessmodel.

---

## 1. State variables

Three timescales / three layers. All per-branch index `b`, spine index `i`.

```
Z_t = { x_b(t),  s_i(t),  M_b(t),  E_b(t),  P_b(t) }_{b,i}
```

| Symbol | Name | Timescale | Range | Biological substrate |
|--------|------|-----------|-------|----------------------|
| `x_b` | branch activation | fast (ms) | [0,1] | dendritic integration / depolarisation |
| `s_i` | spine local access | fast | [0,1] | AMPAR surface density, spine head Ca²⁺ |
| `M_b` | structural accessibility | slow (h–days) | [0, M_max] | cytoskeletal stability, F-actin, spine morphology |
| `E_b` | eligibility trace | intermediate (s–min) | [0,1] | CaMKII activation, synaptic tag |
| `P_b` | translation readiness | intermediate | [0,1] | local mRNA/protein availability (BDNF, Arc, PSD-95) |

Derived per branch:

```
fast_access_b  = σ( g_fast·x_b + g_ctx·bias_b + g_spine·Σ_i(s_i·c_i) − g_inh·inh_b )
slow_access_b  = σ( g_struct · M_b )
A_b            = fast_access_b · slow_access_b          ∈ [0,1]   (effective access)
activation_b   = A_b · cue_b
```

where `σ(·) = 1/(1+e^{-·})` and `g_*` are global gain parameters (`DynamicsParameters`).

---

## 2. Fast timescale — `apply_cue`

One forward pass per presented pattern. Cue input vector `u ∈ ℝ^B`:

```
fast_term_b  = g_fast·u_b + g_ctx·bias_b + g_spine·spine_support_b − g_inh·inh_b
fast_access_b = σ(fast_term_b)
slow_access_b = σ(g_struct · M_b)
A_b           = fast_access_b · slow_access_b
activation_b  = A_b · u_b

E_b ← clamp₀₁[ (1 − λ_E) · E_b + |activation_b| ]     (eligibility accumulation)
P_b ← clamp₀₁[ (1 − λ_P) · P_b ]                       (passive decay between events)
```

Parameters: `g_fast, g_ctx, g_spine, g_inh, g_struct, λ_E (eligibility_decay), λ_P (translation_decay)`.

---

## 3. Slow timescale — `run_consolidation`

Two-pass offline process (replay then sleep). Runs `n_passes` consolidation windows.

### Pass 1 — Translation readiness update (P_b)

For each replayed trace `μ` with priority `p_μ`:

```
replay_contribution_b = Σ_μ [ p_μ · a^μ_b · g_replay ]
sleep_contribution_b  = g_sleep · M_b                     (homeostatic scaling)

P_b ← clamp₀₁[ P_b + replay_contribution_b + sleep_contribution_b ]
```

### Pass 2 — Structural accessibility update (M_b)

Core write rule (bounded, tag-gated, noisy):

```
ΔM_b = η · E_b · P_b · σ(δ_b − θ_δ) · (1 − M_b / M_max)
       − λ_M · M_b
       + √(2 T_eff) · ξ_b
```

where:
- `η` = `structural_lr` — learning rate
- `E_b` — eligibility (synaptic tag)
- `P_b` — local protein/transport support (capture gate)
- `δ_b` = `replay_overlap_b + sleep_contribution_b` — instructional signal
- `θ_δ` — write threshold
- `(1 − M_b/M_max)` — bounded capacity (no over-consolidation)
- `λ_M` — structural decay rate
- `ξ_b ~ N(0,1)`, `T_eff` — structural noise

**Canonical parameter sets** (two experimental contexts):

| Parameter | exp001 / demo | exp013 / canonical (S2 App.) |
|-----------|--------------|------------------------------|
| `structural_gain` | 4.0 | 6.0 |
| `structural_lr` η | 0.20 | 0.18 |
| `eligibility_decay` λ_E | 0.30 | 0.25 |
| `replay_gain` g_replay | 1.20 | 0.80 |
| `sleep_gain` g_sleep | 0.60 | 0.50 |
| consolidation passes | 3 | 9 |

---

## 4. Recall support and readout

For engram trace `μ` with branch weights `{a^μ_b}`:

```
R_μ(t) = Σ_b  a^μ_b · activation_b(t)

expressed_strength_μ = σ( g_readout · (R_μ − θ_readout) )
```

Context mismatch penalty:  
If active context ≠ trace context:  
```
R_μ ← R_μ · max(0, 1 − penalty_ctx)
```

**Branch-linking metric** (used in Article Table 2):

```
M_b1(t) = structural_accessibility of overlap branch b1 after consolidation
ΔM_b1   = M_b1(post) − M_b1(pre)
linking% = 100 · (R_b1(post) − R_b1(pre)) / R_b1(pre)
```

Canonical demo result (exp001 parameters, 3 passes):  
`ΔM_b1 ≈ +0.218`, linking ≈ `+43.1%`.

---

## 5. Mesoscopic export and NDT coupling

### 5.1 Export map

Branch microstate exported to a lower-dimensional accessibility chart:

```
Ψ: Z_t → x^mem_t ∈ ℝ^d
```

On that chart, local flow and Jacobian follow NDT conventions:

```
ẋ^mem = f^mem(x^mem, t) + ε_t
J^mem = ∂f^mem/∂x
Σ^(h+1) = A_t Σ^(h) A_t^⊤ + Q_t          (reachability propagation)
```

### 5.2 Metric and potential coupling

Let P = projection from branch accessibility geometry to regional MNPS chart.

```
G_MNPS(X, Z) ≈ P^⊤ G_access(Ψ(Z)) P

Φ(X, t, Z) = Φ₀(X, t) + Φ_access(X, t, Ψ(Z))
```

Block-wise (regions r,s; chart axes α,β; branch indices a,b):

```
G^block(r,s,α,β; X,Z) ≈ Σ_{a,b} P_r(a,α) · G^access(a,b; Ψ(Z)) · P_s(b,β)
```

Jacobian bridge from micro to macro:

```
J^block(r,s,α,β) ≈ − Σ_{u,γ} G^block(r,u,α,γ) · H(u,s,γ,β)
```

where `H = ∂²Φ/∂x^(u)_γ ∂x^(s)_β` is the potential Hessian.

**Mechanism of action (compact):**  
Slow `M_b` changes → alter `G_access` → propagate through `G_MNPS` → appear as
shifted MNJ blocks, reachability cone volumes, and C/TI/P regime signatures.

---

## 6. qMRI correspondence

The structural accessibility field `M_b` and its support variables `E_b`, `P_b`
are not directly observable, but they have well-defined biological substrates that
qMRI modalities can proxy or constrain.

### 6.1 Variable ↔ qMRI mapping

| Cytodend variable | Biological substrate | qMRI proxy | Metric |
|-------------------|---------------------|------------|--------|
| `M_b` (structural accessibility) | F-actin/cytoskeleton stability, spine density, myeloarchitecture | Quantitative T1 (VFA, MP2RAGE), MTsat, MWF (multi-echo T2\*), NODDI NDI | R1=1/T1 ↑ with myelin; NDI ↑ with neurite density |
| `M_b` slow decay λ_M | Structural turnover rate | Longitudinal ΔR1, ΔMTsat | Training/plasticity studies |
| `P_b` (translation readiness) | BDNF, Arc, PSD-95, local protein synthesis | ¹H-MRS: NAA/Cr (neuroaxonal integrity), Cho/Cr (membrane turnover), total Cr | ↑Cho → active membrane remodelling |
| `E_b` (eligibility trace) | CaMKII, Ca²⁺/calmodulin, synaptic tag duration | Not directly; ASL-CBF as hemodynamic proxy for recent activity | Elevated CBF in recently active regions |
| `s_i.calcium_proxy` | Postsynaptic Ca²⁺ transients, NMDAR occupancy | DKI mean kurtosis (MK), Kapp | Higher MK → denser, more heterogeneous microenvironment |
| `effective_access = fast·slow` | Structural gating of synaptic conductance | T2\* (iron deposition, myeloarchitecture), R2\* | R2\* ↑ with iron/myelin |
| `G_access(Ψ(Z))` (accessibility geometry) | Regional dendritic field organization | DTI: FA, MD, RD, AD; NODDI ODI | Low ODI + high NDI → coherent dense neurite field |
| `Φ_access` (potential contribution) | Structural bias on effective energy landscape | MTR, ihMT (intra-axonal myelin specificity) | ihMT sensitive to myelin bilayers |
| `J^block` (MNJ blocks, post-bridge) | Interregional dendritic field coupling | Structural connectivity (dMRI tractography): streamline density, AFD | Indirect; needs functional coupling validation |

### 6.2 Forward model (schematic)

```
qMRI_b(t) = h_q( M_b(t) ) + η_q

h_q: monotone link function (modality-specific)

Examples:
  R1_b(t) ≈ α_R1 · M_b(t) + β_R1 + ε           (T1 inversion)
  NDI_b(t) ≈ α_NDI · M_b(t)^γ + ε               (NODDI, nonlinear)
  NAA_b(t) ≈ α_NAA · P_b(t) + β_NAA + ε         (MRS proxy for P_b)
```

The forward model is speculative — empirical calibration required.  
Key testable prediction: **longitudinal changes in R1/MTsat in learning-activated  
regions should co-vary with model-predicted ΔM_b** after controlling for  
vascular and non-structural T1 sources.

### 6.3 qMRI → NDT pathway (hypothesis chain)

```
ΔR1(r,t)   }                    ΔM_b(t)             ΔG_access
ΔNDI(r,t)  } → inverse h_q → { ΔP_b(t)    →  Ψ  → ΔΦ_access  →  ΔMNPS / ΔMNJ / ΔCone
ΔCho(r,t)  }                    ΔE_b(t)             ΔJ_block
```

This chain is the core testable hypothesis for qMRI–NDT integration:  
qMRI microstructure metrics constrain priors on slow cytodend variables,  
which in turn shift the accessibility geometry exported to the NDT manifold.

---

## 7. Parameter contracts (DynamicsParameters)

```python
@dataclass
class DynamicsParameters:
    structural_gain:     float = 4.0   # g_struct: slope of slow access sigmoid
    structural_lr:       float = 0.20  # η: M_b write rate
    structural_max:      float = 1.0   # M_max: capacity ceiling
    structural_decay:    float = 0.01  # λ_M: passive turnover
    structural_noise:    float = 0.0   # T_eff: thermal-like noise
    fast_gain:           float = 2.0   # g_fast
    context_gain:        float = 1.0   # g_ctx
    spine_gain:          float = 0.5   # g_spine
    inhibition_gain:     float = 1.0   # g_inh
    eligibility_decay:   float = 0.30  # λ_E
    translation_decay:   float = 0.05  # λ_P
    replay_gain:         float = 1.20  # g_replay
    sleep_gain:          float = 0.60  # g_sleep
    readout_gain:        float = 3.0   # g_readout
    readout_threshold:   float = 0.3   # θ_readout
    context_mismatch_penalty: float = 0.5
```

---

## 8. Boundary conditions and scope

- Model is **minimal**: 4 branches + 1 overlap, no recurrent connectivity.  
  Mechanistic signatures of slow structural writing are isolated from routing artefacts.
- `M_b` update rule is **phenomenological**, not a biochemical rate law.  
  It captures selective stabilisation + bounded capacity without specifying molecular identity.
- The `Ψ: Z_t → x^mem_t` export is **speculative** until branch-resolved perturbation  
  data directly link `ΔZ_t` / `ΔM_b` to `ΔMNPS` / `ΔMNJ` shifts.
- qMRI bridges are **monotone links**, not identity claims.  
  Multiple latent parameterisations can produce observationally similar qMRI patterns.

---

## 9. NDT layer dependency (compact)

```
Micro:  Z_t = {x_b, s_i, M_b, E_b, P_b}                   ← cytodend model
              ↓  Ψ (export map)
Meso:   x^mem_t ∈ ℝ^d  (accessibility geometry)
              ↓  P (regional projection)
Chart:  x_t = [m_t, d_t, e_t]^T  or  x^(9)_t              ← MNPS / Stratified MNPS
              ↓
Flow:   ẋ = f(x,t) = −G_t(x)∇Φ(x,t),  J = ∂f/∂x          ← MNJ
              ↓
Reach:  Σ^(h+1) = A_t Σ^(h) A_t^⊤ + Q_t                  ← Reachability cones
              ↓
Regime: R^{C/TI/P}_t = (C_t, TI_t, P_t)                    ← C/TI/P geometry
```

qMRI enters at the micro layer as an observational constraint on slow variables
`{M_b, P_b}` and propagates upward through the export and coupling chain.

---

*References: cytodendaccessmodel repo (NoeticDiffusion/cytodendaccessmodel);  
NDT Compendium v0.11 (Langell 2026); S1–S3 Appendices (eLife submission).*
