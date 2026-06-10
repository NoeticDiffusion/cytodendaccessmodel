# Simulator Bootstrap CI — Model Anchor

n_repeats=100  CI alpha=0.05 (95%)

| Scenario | LI mean | LI [lo, hi] | CM mean | CM [lo, hi] |
|---|---|---|---|---|
| full_model | 0.6120 | [0.5317, 0.6986] | 0.1920 | [0.1852, 0.1971] |
| fast_context_only | 0.3763 | [0.3276, 0.4265] | 0.1561 | [0.1478, 0.1650] |
| replay_no_structure | 0.4075 | [0.3550, 0.4663] | 0.1561 | [0.1478, 0.1650] |
| random_slow_drift | 0.4065 | [0.3274, 0.4852] | 0.1995 | [0.1970, 0.2004] |
| fixed_allocation_only | 0.4075 | [0.3550, 0.4663] | 0.1995 | [0.1971, 0.2004] |
