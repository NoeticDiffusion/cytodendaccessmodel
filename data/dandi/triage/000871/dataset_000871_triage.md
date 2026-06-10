# Triage Summary: DANDI 000871

## Rationale

- DANDI 000871 is treated as the exploratory dendritic-coupling dataset for the lock-side plausibility program.
- The first-pass selection favors canonical `image+ophys` NWB assets and excludes `raw-movies` and `denoised-movies` derivatives from the initial bundle.
- The candidate bundle is designed around one proof-of-access file plus two same-subject same-session pairs that are useful for early coupling comparisons across planes/depths.

## Selected Assets

| Path | Subject | Session | Size (bytes) | Role | Score | Reasons |
| --- | --- | --- | ---: | --- | ---: | --- |
| `sub-656228/sub-656228_ses-1245548523-acq-1245937736_image+ophys.nwb` | `656228` | `1245548523-acq-1245937736_image+ophys.nwb` | 4043728529 | `proof_of_access` | 132 | canonical_image_ophys, preferred_open_scope_path, paired_session_candidate, image_ophys, subject_tag, session_tag |
| `sub-644972/sub-644972_ses-1237338784-acq-1237809219_image+ophys.nwb` | `644972` | `1237338784-acq-1237809219_image+ophys.nwb` | 1144934234 | `pair_a` | 132 | canonical_image_ophys, preferred_open_scope_path, paired_session_candidate, image_ophys, subject_tag, session_tag |
| `sub-644972/sub-644972_ses-1237338784-acq-1237809217_image+ophys.nwb` | `644972` | `1237338784-acq-1237809217_image+ophys.nwb` | 1079474009 | `pair_a` | 132 | canonical_image_ophys, preferred_open_scope_path, paired_session_candidate, image_ophys, subject_tag, session_tag |
| `sub-656228/sub-656228_ses-1247233186-acq-1247385130_image+ophys.nwb` | `656228` | `1247233186-acq-1247385130_image+ophys.nwb` | 4041517083 | `pair_b` | 132 | canonical_image_ophys, preferred_open_scope_path, paired_session_candidate, image_ophys, subject_tag, session_tag |
| `sub-656228/sub-656228_ses-1247233186-acq-1247385128_image+ophys.nwb` | `656228` | `1247233186-acq-1247385128_image+ophys.nwb` | 4016954448 | `pair_b` | 132 | canonical_image_ophys, preferred_open_scope_path, paired_session_candidate, image_ophys, subject_tag, session_tag |

## Next Step

Probe the selected NWB bundle first, then choose one proof-of-access file and one same-session pair for the first coupling-oriented smoke tests.
