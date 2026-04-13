"""
Article 2 Integrity Check — Table 2 / Table 3 Verification

Replicates full_ablation_benchmark.py with gate_freeze_after=3 for ALL conditions,
matching the V2 ablation benchmark's harmonized setup.

Original benchmark (full_ablation_benchmark.py) used:
  build_l1()       -> gate_freeze_after=4
  build_l2a_only() -> gate_freeze_after=3  <-- CONFOUND

This script uses gate_freeze_after=3 everywhere. It reports:
  - Original Article 2 Table 2 values (hardcoded for reference)
  - Harmonized values from this run
  - Delta per condition (harmonized - original)

If +0.304 / +0.150 L2a advantage disappears -> confound confirmed, Article 2 needs revision.
If +0.304 / +0.150 persist -> freeze timing is not the explanation, investigate further.

Also runs the structured-pattern version (Table 3).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from asomemm.api import PLAsoMemm
from asomemm.runtime import derive_seed_sequence, normalize_seed

BASE_SEED = 20260329
N_SEEDS = 5
CORRUPTION = 0.35
DIM = 256
N_ACTIVE = 64
N_FINE = 32
CTX_COUNT = 4

SEED_SEQ = derive_seed_sequence(normalize_seed(BASE_SEED), N_SEEDS, namespace="report_seed")

# --- Original Article 2 Table 2 values (random patterns, 35% corruption) ---
TABLE2_ORIG = {
    #  P:  (L1,    L2b,   L2a)
    6:  (0.611, 0.615, 0.915),
    10: (0.501, 0.492, 0.651),
    14: (0.581, 0.571, 0.603),
    18: (0.402, 0.395, 0.385),
    24: (0.257, 0.252, 0.206),
}

# --- Original Article 2 Table 3 values (structured patterns, 35% corruption) ---
TABLE3_ORIG = {
    #  P:  (L1,    L2a)
    6:  (0.960, 0.910),
    10: (0.808, 0.796),
    14: (0.788, 0.685),
    18: (0.752, 0.618),
    24: (0.738, 0.581),
}


# ─── builders — ALL with gate_freeze_after=3 ─────────────────────────────────

def build_l1_harmonized(settle_steps: int = 15) -> PLAsoMemm:
    return PLAsoMemm.build_pseudo_likelihood_baseline(
        dim=DIM, gated=True, gate_mode="dataset", gate_freeze_after=3,
        settle_steps=settle_steps, gain=1.5, lr=0.05,
        epochs_per_consolidation=200, consolidation_mode="autonomous",
    )


def build_l2b_harmonized(settle_steps: int = 15) -> PLAsoMemm:
    return PLAsoMemm.build_two_level_l2b_only(
        dim=DIM, gain_alpha=1.0, gate_freeze_after=3,
        settle_steps=settle_steps, gain=1.5,
        lr=0.05, epochs_per_consolidation=200, consolidation_mode="autonomous",
    )


def build_l2a_harmonized(settle_steps: int = 15) -> PLAsoMemm:
    return PLAsoMemm.build_two_level_baseline(
        dim=DIM, n_active=N_ACTIVE, n_fine=N_FINE, gain_alpha=1.0,
        use_gain_modulation=False, gate_freeze_after=3, subfield_freeze_after=6,
        settle_steps=settle_steps, gain=1.5, lr=0.05,
        epochs_per_consolidation=200,
    )


# ─── helpers ─────────────────────────────────────────────────────────────────

def _bipolar(D: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=D).astype(np.float32) * 2 - 1


def _structured(D: int, n_active: int, rng: np.random.Generator) -> tuple[list[int], np.ndarray]:
    stable = sorted(rng.choice(D, n_active, replace=False).tolist())
    proto = _bipolar(D, rng)
    return stable, proto


def _flip(pat: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    c = pat.copy()
    n = int(len(c) * frac)
    c[rng.choice(len(c), n, replace=False)] *= -1
    return c


def _csim(a: np.ndarray, b: np.ndarray, slots: list[int]) -> float:
    av, bv = a[np.asarray(slots)], b[np.asarray(slots)]
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    return float(np.dot(av, bv) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0


def _run_one(
    seed: int, P: int, corrupt: float, builder,
    pattern_type: str = "random", settle_steps: int = 15,
) -> float:
    rng = np.random.default_rng(seed)
    contexts = [f"ctx{i}" for i in range(CTX_COUNT)]
    system = builder(settle_steps)

    if pattern_type == "random":
        pats = {ctx: [_bipolar(DIM, rng) for _ in range(P)] for ctx in contexts}
    else:
        # structured: each context gets its own stable slots + prototype
        pats = {}
        for ctx in contexts:
            stable, proto = _structured(DIM, N_ACTIVE, rng)
            ctx_pats = []
            for _ in range(P):
                p = _bipolar(DIM, rng)
                p[stable] = proto[stable]
                ctx_pats.append(p)
            pats[ctx] = ctx_pats

    for ctx in contexts:
        for pat in pats[ctx]:
            system.encode(pat, context=ctx)
    system.consolidate_now()

    frozen_masks = getattr(
        getattr(system.gating, "base_gate", system.gating), "_frozen_masks", {}
    )
    tgt_ctx = contexts[0]
    tgt_mask = frozen_masks.get(tgt_ctx)
    slots = list(tgt_mask.active_slots) if tgt_mask else list(range(N_ACTIVE))

    sims = []
    for pat in pats[tgt_ctx]:
        cue = _flip(pat, corrupt, rng)
        result = system.recall(cue, context=tgt_ctx)
        sims.append(_csim(result.recalled, pat, slots))
    return float(np.mean(sims))


def _run_condition(
    P: int, corrupt: float, builder, pattern_type: str = "random"
) -> tuple[float, float]:
    vals = [_run_one(int(s), P, corrupt, builder, pattern_type) for s in SEED_SEQ]
    return float(np.mean(vals)), float(np.std(vals))


# ─── main verification ────────────────────────────────────────────────────────

def verify_table2() -> dict:
    """Replicate Table 2 (random patterns) with harmonized gate_freeze_after=3."""
    P_LIST = [6, 10, 14, 18, 24]
    print(f"\n{'='*90}")
    print(f"TABLE 2 VERIFICATION — Random patterns, 35% corruption, 5 seeds")
    print(f"All conditions: gate_freeze_after=3 (harmonized)")
    print(f"{'='*90}")
    print(f"{'P':>4}  {'alpha':>5}  "
          f"{'L1_orig':>8}  {'L1_new':>8}  {'dL1':>6}  "
          f"{'L2a_orig':>9}  {'L2a_new':>8}  {'dL2a':>7}  "
          f"{'Delta_orig':>11}  {'Delta_new':>10}  verdict")
    print("-" * 100)

    results = {}
    for P in P_LIST:
        alpha = P / N_ACTIVE
        l1_orig, l2a_orig = TABLE2_ORIG[P][0], TABLE2_ORIG[P][2]

        l1_new, l1_std   = _run_condition(P, CORRUPTION, build_l1_harmonized, "random")
        l2a_new, l2a_std = _run_condition(P, CORRUPTION, build_l2a_harmonized, "random")

        d_l1   = l1_new  - l1_orig
        d_l2a  = l2a_new - l2a_orig
        delta_orig = l2a_orig - l1_orig
        delta_new  = l2a_new  - l1_new

        verdict = "ARTEFACT" if abs(delta_new) < 0.05 and abs(delta_orig) > 0.10 else (
                  "REDUCED"  if delta_new < delta_orig - 0.05 else
                  "UNCHANGED")

        print(f"{P:>4}  {alpha:.3f}  "
              f"{l1_orig:>8.3f}  {l1_new:>8.3f}  {d_l1:>+6.3f}  "
              f"{l2a_orig:>9.3f}  {l2a_new:>8.3f}  {d_l2a:>+7.3f}  "
              f"{delta_orig:>+11.3f}  {delta_new:>+10.3f}  {verdict}")

        results[P] = {
            "l1_orig": l1_orig, "l1_new": l1_new,
            "l2a_orig": l2a_orig, "l2a_new": l2a_new,
            "delta_orig": delta_orig, "delta_new": delta_new,
        }
    return results


def verify_table3() -> dict:
    """Replicate Table 3 (structured patterns) with harmonized gate_freeze_after=3."""
    P_LIST = [6, 10, 14, 18, 24]
    print(f"\n{'='*90}")
    print(f"TABLE 3 VERIFICATION — Structured patterns, 35% corruption, 5 seeds")
    print(f"All conditions: gate_freeze_after=3 (harmonized)")
    print(f"{'='*90}")
    print(f"{'P':>4}  {'alpha':>5}  "
          f"{'L1_orig':>8}  {'L1_new':>8}  {'dL1':>6}  "
          f"{'L2a_orig':>9}  {'L2a_new':>8}  {'dL2a':>7}  "
          f"{'Delta_orig':>11}  {'Delta_new':>10}  verdict")
    print("-" * 100)

    results = {}
    for P in P_LIST:
        alpha = P / N_ACTIVE
        l1_orig, l2a_orig = TABLE3_ORIG[P]

        l1_new, l1_std   = _run_condition(P, CORRUPTION, build_l1_harmonized, "structured")
        l2a_new, l2a_std = _run_condition(P, CORRUPTION, build_l2a_harmonized, "structured")

        d_l1   = l1_new  - l1_orig
        d_l2a  = l2a_new - l2a_orig
        delta_orig = l2a_orig - l1_orig
        delta_new  = l2a_new  - l1_new

        verdict = "ARTEFACT" if delta_orig < 0 and abs(delta_new) < 0.03 else (
                  "REDUCED"  if abs(delta_new) < abs(delta_orig) * 0.6 else
                  "UNCHANGED")

        print(f"{P:>4}  {alpha:.3f}  "
              f"{l1_orig:>8.3f}  {l1_new:>8.3f}  {d_l1:>+6.3f}  "
              f"{l2a_orig:>9.3f}  {l2a_new:>8.3f}  {d_l2a:>+7.3f}  "
              f"{delta_orig:>+11.3f}  {delta_new:>+10.3f}  {verdict}")

        results[P] = {
            "l1_orig": l1_orig, "l1_new": l1_new,
            "l2a_orig": l2a_orig, "l2a_new": l2a_new,
            "delta_orig": delta_orig, "delta_new": delta_new,
        }
    return results


if __name__ == "__main__":
    print("#" * 90)
    print("# ARTICLE 2 INTEGRITY CHECK")
    print("# Hypothesis: original L2a gains are freeze-timing confounds (gate_freeze_after 3 vs 4)")
    print("#" * 90)

    r2 = verify_table2()
    r3 = verify_table3()

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    p6  = r2.get(6,  {})
    p10 = r2.get(10, {})
    print(f"Random P=6:  delta_orig={p6.get('delta_orig', 'N/A'):+.3f}  "
          f"delta_new={p6.get('delta_new', 'N/A'):+.3f}")
    print(f"Random P=10: delta_orig={p10.get('delta_orig', 'N/A'):+.3f}  "
          f"delta_new={p10.get('delta_new', 'N/A'):+.3f}")
    t3_p10 = r3.get(10, {})
    t3_p18 = r3.get(18, {})
    print(f"Struct P=10: delta_orig={t3_p10.get('delta_orig', 'N/A'):+.3f}  "
          f"delta_new={t3_p10.get('delta_new', 'N/A'):+.3f}")
    print(f"Struct P=18: delta_orig={t3_p18.get('delta_orig', 'N/A'):+.3f}  "
          f"delta_new={t3_p18.get('delta_new', 'N/A'):+.3f}")

    if (abs(p6.get('delta_new', 9)) < 0.05 and abs(p10.get('delta_new', 9)) < 0.05):
        print("\nVERDICT: CONFOUND CONFIRMED")
        print("  The L2a gains in Article 2 Table 2 appear to be freeze-timing artefacts.")
        print("  Article 2 abstract, title framing, and tables require revision.")
    elif (p6.get('delta_new', 0) > 0.15 and p10.get('delta_new', 0) > 0.05):
        print("\nVERDICT: GAINS PERSIST — freeze timing is NOT the explanation")
        print("  Investigate other sources of discrepancy (pattern generator, context count, etc.).")
    else:
        print("\nVERDICT: MIXED — partial reduction")
        print("  Some but not all of the gain is explained by freeze timing. Needs investigation.")
