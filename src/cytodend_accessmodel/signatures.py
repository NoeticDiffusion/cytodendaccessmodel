"""Canonical signature computation for the branch-resolved cytoskeletal-dendritic
accessibility model.

This module is the single shared source of truth for SIG-A to SIG-E.
All experiment scripts (E017, E018, E019, E020 …) should derive their signature
scores from :func:`compute_signature_profile` rather than re-implementing the
formulas independently.

Signature definitions
---------------------
SIG-A  Overlap-branch selective structural writing
        ``delta_M_b(overlap) - mean(delta_M_b(non-overlap))``
        Units: ΔM_b (dimensionless, bounded ≈ [-1, 1])

SIG-B  Linking gain after consolidation
        ``L_post_consolidation - L_pre_consolidation``
        Units: ΔL (dimensionless, bounded by allocation geometry)

SIG-C  Context separation (architectural fast-gating signature)
        ``mean(correct_recall - wrong_recall)`` across alpha/beta probe
        Units: recall-support units (dimensionless)
        Diagnostic status: architectural / auxiliary — reflects allocation
        geometry, not slow structural writing.

SIG-D  Linking > recall dissociation under overlap damage
        ``(link_drop_pct) - (recall_drop_pct)``
        Units: percentage points (ΔL / L_healthy and Δrecall / recall_healthy
        are both bounded ratios × 100)
        Diagnostic status: partly driven by overlap-branch perturbation geometry;
        not diagnostic alone.

SIG-E  Targeted rescue selectivity
        Raw:        ``L_targeted_rescue - L_reference_rescue``
        Normalized: ``NR_targeted - NR_reference``
                    where NR = (L_post_rescue − L_post_damage) /
                                 (L_healthy    − L_post_damage)
        Units: raw = ΔL; normalized = dimensionless ratio (can exceed 1.0
        when rescue overshoots the healthy reference).
        NOTE: do NOT label SIG-E in "percentage points" — L is unbounded above
        the pre-damage healthy level when consolidation overshoots.
        Diagnostic status: rescue-selectivity signature, protocol-sensitive.
        Magnitude depends on whether pre-rescue probe cues are present.

Protected thresholds (predeclared, identical across E017–E020)
--------------------------------------------------------------
SIG-A: delta_M_b > 0.02
SIG-B: delta_L   > 0.05
SIG-C: separation> 0.05
SIG-D: pp diff   > 5.0
SIG-E: NR diff   > 0.10   (normalized) OR raw diff > 0.02

Protocol notes (E019R canonical forward protocol)
--------------------------------------------------
The canonical protocol for E020 and later experiments is the E019 variant:
  encode → consolidate → damage → rescue
WITHOUT post-consolidation or post-damage probe cues between phases.
This gives the maximal SIG-E dynamic range (generic rescue baseline is clean).
E017 and E018 used probe cues between phases, which suppressed SIG-E to ~33pp
by pre-loading E_b on all branches before the rescue comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Predeclared thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, float] = {
    "SIG_A": 0.02,   # delta M_b
    "SIG_B": 0.05,   # delta L
    "SIG_C": 0.05,   # support units
    "SIG_D": 5.0,    # percentage points
    "SIG_E_normalized": 0.10,   # normalized recovery difference
    "SIG_E_raw":       0.02,    # raw linking difference
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RescueConditionResult:
    """Result for one rescue condition in the SIG-E audit."""
    name: str
    L_post_rescue: float
    absolute_recovery: float = 0.0          # L_post_rescue - L_post_damage
    normalized_recovery: float = 0.0        # absolute_recovery / (L_healthy - L_post_damage)
    overshoot: bool = False                  # normalized_recovery > 1.0

    @classmethod
    def compute(
        cls, name: str, L_post_rescue: float,
        L_post_damage: float, L_healthy: float,
    ) -> "RescueConditionResult":
        abs_rec = L_post_rescue - L_post_damage
        denom   = L_healthy - L_post_damage
        nr = abs_rec / denom if abs(denom) > 1e-9 else 0.0
        return cls(
            name=name,
            L_post_rescue=L_post_rescue,
            absolute_recovery=abs_rec,
            normalized_recovery=nr,
            overshoot=nr > 1.0,
        )


@dataclass
class SignatureInputs:
    """All intermediate values required to compute SIG-A through SIG-E.

    Collect these from your simulation run and pass to
    :func:`compute_signature_profile`.
    """
    # SIG-A
    mb_overlap_pre: float
    mb_overlap_post_cons: float
    mb_nonoverlap_mean_pre: float
    mb_nonoverlap_mean_post_cons: float

    # SIG-B
    L_pre: float
    L_post_cons: float

    # SIG-C (from dedicated context-probe simulation)
    context_separation: float

    # SIG-D
    L_post_damage: float
    recall_support_post_cons: float
    recall_support_post_damage: float

    # SIG-E: two rescue conditions must be provided
    rescue_conditions: list[RescueConditionResult] = field(default_factory=list)
    targeted_rescue_name: str = "targeted_overlap"
    reference_rescue_name: str = "generic_plain_consolidation"

    # Protocol metadata (for auditing)
    protocol_name: str = "canonical_e019"
    notes: str = ""


@dataclass
class SignatureProfile:
    """Computed signature profile."""
    SIG_A: float
    SIG_B: float
    SIG_C: float
    SIG_D: float                    # pp (linking% drop - recall% drop)
    SIG_E_raw: float                # ΔL (targeted - reference)
    SIG_E_normalized: float         # ΔNR (can exceed 1.0 = overshoot)
    SIG_E_targeted_overshoot: bool
    SIG_E_reference_overshoot: bool

    thresholds: dict[str, float]
    directional_passes: dict[str, bool]
    protected_passes: dict[str, bool]
    joint_protected_pass: bool

    # Raw diagnostics
    L_pre: float = 0.0
    L_post_cons: float = 0.0
    L_post_damage: float = 0.0
    rescue_results: list[RescueConditionResult] = field(default_factory=list)

    protocol_name: str = "canonical_e019"
    notes: str = ""


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_signature_profile(
    inputs: SignatureInputs,
    thresholds: Optional[dict[str, float]] = None,
) -> SignatureProfile:
    """Compute SIG-A through SIG-E from pre-computed simulation intermediates.

    Parameters
    ----------
    inputs:
        :class:`SignatureInputs` with all required intermediate values.
    thresholds:
        Override default protected thresholds.  Must contain keys
        ``SIG_A``, ``SIG_B``, ``SIG_C``, ``SIG_D``,
        ``SIG_E_normalized``, ``SIG_E_raw``.

    Returns
    -------
    :class:`SignatureProfile`
    """
    thr = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        thr.update(thresholds)

    # SIG-A
    sig_a = (
        (inputs.mb_overlap_post_cons    - inputs.mb_overlap_pre) -
        (inputs.mb_nonoverlap_mean_post_cons - inputs.mb_nonoverlap_mean_pre)
    )

    # SIG-B
    sig_b = inputs.L_post_cons - inputs.L_pre

    # SIG-C
    sig_c = inputs.context_separation

    # SIG-D: percentage-point difference in damage sensitivity
    lk_drop_pct  = _pct_drop(inputs.L_post_cons,            inputs.L_post_damage)
    rec_drop_pct = _pct_drop(inputs.recall_support_post_cons, inputs.recall_support_post_damage)
    sig_d = lk_drop_pct - rec_drop_pct

    # SIG-E
    L_healthy = inputs.L_post_cons  # healthy reference = post-consolidation linking
    L_damaged = inputs.L_post_damage

    targ = _find_rescue(inputs.rescue_conditions, inputs.targeted_rescue_name)
    ref  = _find_rescue(inputs.rescue_conditions, inputs.reference_rescue_name)

    if targ is None or ref is None:
        sig_e_raw = float("nan")
        sig_e_norm = float("nan")
        targ_overshoot = False
        ref_overshoot  = False
    else:
        sig_e_raw  = targ.L_post_rescue  - ref.L_post_rescue
        sig_e_norm = targ.normalized_recovery - ref.normalized_recovery
        targ_overshoot = targ.overshoot
        ref_overshoot  = ref.overshoot

    # Pass / fail
    dir_passes = {
        "SIG_A": sig_a > 0,
        "SIG_B": sig_b > 0,
        "SIG_C": sig_c > 0,
        "SIG_D": sig_d > 0,
        "SIG_E": (sig_e_norm > 0) if not _is_nan(sig_e_norm) else False,
    }
    prot_passes = {
        "SIG_A": sig_a > thr["SIG_A"],
        "SIG_B": sig_b > thr["SIG_B"],
        "SIG_C": sig_c > thr["SIG_C"],
        "SIG_D": sig_d > thr["SIG_D"],
        "SIG_E": (sig_e_norm > thr["SIG_E_normalized"]) if not _is_nan(sig_e_norm) else False,
    }

    return SignatureProfile(
        SIG_A=sig_a,
        SIG_B=sig_b,
        SIG_C=sig_c,
        SIG_D=sig_d,
        SIG_E_raw=sig_e_raw,
        SIG_E_normalized=sig_e_norm,
        SIG_E_targeted_overshoot=targ_overshoot,
        SIG_E_reference_overshoot=ref_overshoot,
        thresholds=thr,
        directional_passes=dir_passes,
        protected_passes=prot_passes,
        joint_protected_pass=all(prot_passes.values()),
        L_pre=inputs.L_pre,
        L_post_cons=inputs.L_post_cons,
        L_post_damage=inputs.L_post_damage,
        rescue_results=inputs.rescue_conditions,
        protocol_name=inputs.protocol_name,
        notes=inputs.notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_drop(healthy: float, damaged: float) -> float:
    """Percent drop: (healthy - damaged) / healthy × 100."""
    return (healthy - damaged) / max(abs(healthy), 1e-9) * 100.0


def _find_rescue(
    conditions: list[RescueConditionResult], name: str,
) -> Optional[RescueConditionResult]:
    for c in conditions:
        if c.name == name:
            return c
    return None


def _is_nan(x: float) -> bool:
    import math
    return math.isnan(x)


# ---------------------------------------------------------------------------
# Convenience: build SignatureInputs from common run output dicts
# ---------------------------------------------------------------------------

def inputs_from_run_dict(run: dict, rescue_map: dict[str, float],
                         L_post_damage: float, protocol_name: str = "canonical_e019") -> SignatureInputs:
    """Build :class:`SignatureInputs` from a dict of run outputs.

    Parameters
    ----------
    run:
        Dict with keys:
            mb_pre_overlap, mb_post_overlap,
            mb_nonoverlap_mean_pre, mb_nonoverlap_mean_post,
            L_pre, L_post_cons,
            context_separation,
            recall_support_post_cons, recall_support_post_damage,
    rescue_map:
        ``{condition_name: L_post_rescue}``
    L_post_damage:
        Linking score after damage phase.
    protocol_name:
        Label for the protocol variant.
    """
    L_healthy = run["L_post_cons"]
    rescues = [
        RescueConditionResult.compute(name, L_resc, L_post_damage, L_healthy)
        for name, L_resc in rescue_map.items()
    ]
    return SignatureInputs(
        mb_overlap_pre=run["mb_pre_overlap"],
        mb_overlap_post_cons=run["mb_post_overlap"],
        mb_nonoverlap_mean_pre=run.get("mb_nonoverlap_mean_pre", float("nan")),
        mb_nonoverlap_mean_post_cons=run.get("mb_nonoverlap_mean_post", float("nan")),
        L_pre=run["L_pre"],
        L_post_cons=L_healthy,
        context_separation=run["context_separation"],
        L_post_damage=L_post_damage,
        recall_support_post_cons=run["recall_support_post_cons"],
        recall_support_post_damage=run["recall_support_post_damage"],
        rescue_conditions=rescues,
        protocol_name=protocol_name,
    )
