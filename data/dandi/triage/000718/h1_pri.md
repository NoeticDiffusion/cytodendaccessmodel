# H1 Selective Replay: PRI

top_frac=0.15  activity_threshold=0.0  sigma=2.0

| Pair | Real frac | C1 shuffle | C3 event-shuffle | Verdict |
|---|---|---|---|---|
| Ca-EEG3-4 OffD1 | 0.085 | 0.031 ± 0.009 | 0.122 ± 0.018 | **PARTIAL** |
| Ca-EEG3-4 OffD2 | 0.116 | 0.032 ± 0.014 | 0.106 ± 0.034 | **PARTIAL** |
| Ca-EEG2-1 OffD2 | 0.140 | 0.029 ± 0.010 | 0.096 ± 0.034 | **PARTIAL** |

**Overall: H1 PARTIAL — some selectivity, bridge refinement needed**


## Per-Method PRI (real result)


### Ca-EEG3-4 OffD1

n_registered=621  n_events=32

| Method | n_sig | total | frac_sig | mean_z | max_z |
|---|---|---|---|---|---|
| nmf | 36 | 256 | 0.141 | 0.499 | 8.338 |
| ica | 21 | 256 | 0.082 | 0.287 | 3.491 |
| graph | 8 | 256 | 0.031 | -0.055 | 3.5 |

### Ca-EEG3-4 OffD2

n_registered=686  n_events=9

| Method | n_sig | total | frac_sig | mean_z | max_z |
|---|---|---|---|---|---|
| nmf | 12 | 72 | 0.167 | 0.433 | 5.061 |
| ica | 9 | 72 | 0.125 | 0.128 | 4.156 |
| graph | 4 | 72 | 0.056 | -0.157 | 3.993 |

### Ca-EEG2-1 OffD2

n_registered=485  n_events=14

| Method | n_sig | total | frac_sig | mean_z | max_z |
|---|---|---|---|---|---|
| nmf | 17 | 112 | 0.152 | 0.475 | 5.323 |
| ica | 18 | 112 | 0.161 | 0.566 | 6.277 |
| graph | 12 | 112 | 0.107 | 0.416 | 4.106 |
