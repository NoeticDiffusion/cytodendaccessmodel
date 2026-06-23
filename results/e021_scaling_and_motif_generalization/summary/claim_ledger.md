# E021 — Claim Ledger

| Claim | Status | Evidence | Limitation | Next action |
|---|---|---|---|---|
| Mechanism survives beyond 4-branch/2-trace canonical setup | Validated | 20/24 runs joint-pass | 6 specific motif types | E022 hard comparators |
| Canonical behavior reproduced | Validated | canonical motif passes at all branch counts | Canonical params only | — |
| Strong overlap generalizes canonical mechanism | Validated if strong_overlap passes | generalization_summary_by_motif.csv | 3 branch counts | E022 |
| Chain topology: local > distant linking | gSIG-B + false_linking_rate confirms | chain_overlap runs | Chain only | E022 |
| Hub topology increases false-linking risk | false_linking_rate for hub | false_linking_summary.csv | hub only | E022 |
| Sparse random: linking scales with density | gSIG-B passes at multiple seeds | sparse_random runs | 2 seeds, 3 counts | E022 |
| Model scales to realistic neurons | Pending | Requires beyond 32 branches | — | E022+ |
| Model generalizes to arbitrary memory systems | Not supported | — | — | — |
