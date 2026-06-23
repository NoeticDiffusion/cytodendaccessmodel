# e018 — Context Probe Limitations

SIG-C is computed in a separate context-probe simulation, not the main 10-phase trace.

## Why separate
The main phase trace uses mu1/mu2 (linking traces). SIG-C requires mu_alpha/mu_beta
(context-differentiated traces with distinct branch allocations and explicit context labels).
Merging these into one trace would require a different simulator configuration.

## What is exported
- `traces/<comparator>_context_probe_traces.csv`: per-branch and per-trace snapshots
  for alpha_probe and beta_probe conditions.
- `summary/<comparator>_signature_summary.csv`: SIG-C score included.

## What is NOT exported
- Time-resolved context-probe traces across consolidation phases (only post-consolidation probe).
- Context-value per branch in main branch_traces.csv.

## Impact on claims
SIG-C direction and magnitude are reported. The limitation is that SIG-C dynamics
across consolidation cannot be visualized from main traces. This will be addressed in E019
if context-trace instrumentation is added to the main protocol.
