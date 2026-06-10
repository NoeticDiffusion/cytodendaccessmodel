# H1: Registration-Aware Event-Based Pipeline

WP1 registration: centroid+Dice+shape+neighbourhood confidence

WP2 events: threshold_sigma=2.0

WP3 assemblies: NMF + ICA + Graph  k=8


## Results

| Subject | Pair | Registered units | Verdict |
|---|---|---|---|
| Ca-EEG2-1 | NE->OffD2 | 485 | **positive** |
| Ca-EEG3-4 | NE->OffD1 | 621 | **positive** |
| Ca-EEG3-4 | NE->OffD2 | 686 | **positive** |

### Ca-EEG2-1 NE->OffD2

Registration: {'session_a': 'Ca-EEG2-1_NE', 'session_b': 'Ca-EEG2-1_offline_d2', 'n_rois_a': 623, 'n_rois_b': 625, 'n_candidates': 485, 'n_accepted': 485, 'fraction_a_matched': 0.778, 'fraction_b_matched': 0.776, 'mean_confidence': 0.748, 'mean_dice': 0.861, 'mean_centroid_dist_px': 4.19}

| Method | Events | N sig | Frac sig | Max z | Stab mean |
|---|---|---|---|---|---|
| deconvolved_nmf | 14 | 22 | 0.196 | 8.66 | 1.0 |
| deconvolved_ica | 14 | 47 | 0.42 | 7.412 | 0.997 |
| deconvolved_graph | 14 | 60 | 0.536 | 5.69 | 0.929 |

### Ca-EEG3-4 NE->OffD1

Registration: {'session_a': 'Ca-EEG3-4_NE', 'session_b': 'Ca-EEG3-4_offline_d1', 'n_rois_a': 851, 'n_rois_b': 711, 'n_candidates': 621, 'n_accepted': 621, 'fraction_a_matched': 0.73, 'fraction_b_matched': 0.873, 'mean_confidence': 0.797, 'mean_dice': 0.878, 'mean_centroid_dist_px': 3.06}

| Method | Events | N sig | Frac sig | Max z | Stab mean |
|---|---|---|---|---|---|
| deconvolved_nmf | 32 | 88 | 0.344 | 7.279 | 1.0 |
| deconvolved_ica | 32 | 104 | 0.406 | 6.482 | 0.996 |
| deconvolved_graph | 32 | 124 | 0.484 | 7.407 | 0.951 |

### Ca-EEG3-4 NE->OffD2

Registration: {'session_a': 'Ca-EEG3-4_NE', 'session_b': 'Ca-EEG3-4_offline_d2', 'n_rois_a': 851, 'n_rois_b': 858, 'n_candidates': 686, 'n_accepted': 686, 'fraction_a_matched': 0.806, 'fraction_b_matched': 0.8, 'mean_confidence': 0.771, 'mean_dice': 0.871, 'mean_centroid_dist_px': 3.86}

| Method | Events | N sig | Frac sig | Max z | Stab mean |
|---|---|---|---|---|---|
| deconvolved_nmf | 9 | 25 | 0.347 | 6.173 | 1.0 |
| deconvolved_ica | 9 | 31 | 0.431 | 4.967 | 1.0 |
| deconvolved_graph | 9 | 45 | 0.625 | 3.675 | 0.951 |
