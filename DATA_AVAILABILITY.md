# Data Availability

## Publicly available in this repository

- Source code for the FocusTrace local research platform.
- Preprocessing and dataset-construction scripts.
- Participant-grouped training and evaluation code.
- Experiment and research configuration.
- Aggregate and fold-level evaluation metrics.
- Selected trained model artifacts for the best reported binary experiment.
- Protocol and reproducibility documentation.

## Not publicly released

The following human-participant materials are intentionally excluded:

- raw webcam recordings;
- consent forms and identity linkage;
- participant responses;
- session databases and event logs;
- participant-level extracted features;
- the final participant-level `dataset.npz`;
- out-of-fold prediction files linked to participant codes.

The consent documentation used for this study did not explicitly authorize unrestricted public redistribution of participant-level research data. Removing names alone is not sufficient protection because webcam recordings contain identifiable faces and participant-level behavioral traces may still be sensitive.

Accordingly, the public artifact is designed to support **methodological transparency and code review**, not unrestricted redistribution of the original cohort data.

## Dataset summary reported in the manuscript

The analyzed dataset contains 12,015 quality-screened overlapping windows from 18 participants. Windows are 2 seconds long, use a 1-second stride, and are represented on a 5 Hz grid with 21 final model features.

Final label counts:

- DTE: 3,934 windows
- TREE: 4,084 windows
- TIE: 3,997 windows

Binary mapping:

- On-task (DTE + TREE): 8,018 windows
- Distraction (TIE): 3,997 windows

These counts are reported for transparency but the participant-level rows are not included in the repository.
