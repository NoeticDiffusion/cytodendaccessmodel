# e018 — Effect Summary

## Signature matrix

| Comparator                 | SIG-A (ΔM_b) | SIG-B (ΔL)   | SIG-C (sup)  | SIG-D (pp)   | SIG-E (pp)   | Joint |
|----------------------------|--------------|--------------|--------------|--------------|--------------|-------|
| full_model                 | 0.1281 (P) | 0.2397 (P) | 0.1545 (P) | 20.9827 (P) | 33.2919 (P) | PASS |
| fast_context_only          | 0.0000 (F) | 0.0000 (F) | 0.1490 (P) | 19.2511 (P) | 0.0000 (F) | FAIL |
| replay_no_structure        | 0.0000 (F) | -0.0180 (F) | 0.1480 (P) | 19.2118 (P) | 0.0000 (F) | FAIL |
| random_slow_drift          | -0.0889 (F) | -0.0646 (F) | 0.1486 (P) | -5.1377 (F) | -69.3736 (F) | FAIL |
| fixed_allocation_only      | 0.0000 (F) | 0.0000 (F) | 0.1490 (P) | 19.2511 (P) | 0.0000 (F) | FAIL |

## Predeclared thresholds
SIG-A > 0.02 ΔM_b | SIG-B > 0.05 ΔL |
SIG-C > 0.05 support | SIG-D > 5.0 pp |
SIG-E > 10.0 pp
