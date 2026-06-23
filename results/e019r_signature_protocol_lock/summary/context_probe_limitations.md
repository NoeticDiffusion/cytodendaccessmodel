# Context Probe Limitations (E019R)

SIG-C is computed in a dedicated context-probe simulation, not in the main
ten-phase protocol. The context-probe uses mu_alpha/mu_beta allocations and
explicit context labels; these are incompatible with the main mu1/mu2 linking
protocol.

This limitation was documented in E018 and carries forward. Full time-resolved
context traces across consolidation are deferred to a later experiment.
