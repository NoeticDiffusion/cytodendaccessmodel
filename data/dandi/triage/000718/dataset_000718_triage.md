# Triage Summary: DANDI 000718

## Rationale

- DANDI 000718 is treated as the primary linking dataset because it directly targets offline ensemble co-reactivation and memory integration across days.
- The first-pass selection favors NWB assets with offline/week/session hints, both subjects when available, and metadata useful for later linking analysis triage.
- This adapter stops at manifest and metadata triage; it does not perform ensemble extraction or biological hypothesis testing.

## Selected Assets

| Path | Subject | Session | Size (bytes) | Score | Reasons |
| --- | --- | --- | ---: | ---: | --- |
| `sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-OfflineDay2Session1_ophys.nwb` | `Ca-EEG2-1` | `OfflineDay2Session1_ophys.nwb` | 4674074229 | 31 | offline, reactivation, neutral, aversive, fear, memory, cell, eeg, ophys, subject_tag, session_tag, nwb |
| `sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-OfflineDay1Session1_ophys.nwb` | `Ca-EEG3-4` | `OfflineDay1Session1_ophys.nwb` | 4633762345 | 31 | offline, reactivation, neutral, aversive, fear, memory, cell, eeg, ophys, subject_tag, session_tag, nwb |
| `sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-OfflineDay2Session1_ophys.nwb` | `Ca-EEG3-4` | `OfflineDay2Session1_ophys.nwb` | 4620479171 | 31 | offline, reactivation, neutral, aversive, fear, memory, cell, eeg, ophys, subject_tag, session_tag, nwb |
| `sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-NeutralExposure_image+ophys.nwb` | `Ca-EEG3-4` | `NeutralExposure_image+ophys.nwb` | 4715546595 | 29 | offline, reactivation, neutral, aversive, memory, cell, eeg, ophys, image, subject_tag, session_tag, nwb |
| `sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-NeutralExposure_image+ophys.nwb` | `Ca-EEG2-1` | `NeutralExposure_image+ophys.nwb` | 4664000302 | 29 | offline, reactivation, neutral, aversive, memory, cell, eeg, ophys, image, subject_tag, session_tag, nwb |
| `sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-Recall3_image+ophys.nwb` | `Ca-EEG3-4` | `Recall3_image+ophys.nwb` | 2337241107 | 29 | offline, reactivation, neutral, aversive, memory, cell, eeg, ophys, image, subject_tag, session_tag, nwb |

## Next Step

Use the selected asset bundle for metadata/header probing first, then decide which subset should be downloaded for the first offline-linking analysis pass.
