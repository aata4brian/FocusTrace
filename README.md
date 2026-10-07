# FocusTrace

**FocusTrace** is a webcam-based research framework for classifying observable learning-related behavior while explicitly separating **task relevance** from whether attention is directed toward the laptop screen.

This repository accompanies the 2026 research manuscript:

> **FocusTrace: Integrasi Analisis Perilaku dan Machine Learning untuk Klasifikasi Perilaku Fokus dalam Pembelajaran**  
> Moch. Brian Mursyidan Baldan and Shabrina Putri Mardhita — Universitas Airlangga

## Research scope

FocusTrace does **not** claim to measure a participant's internal mental focus, concentration, motivation, or comprehension. It models observable behavioral patterns under controlled task conditions.

The experimental protocol uses three conditions:

- **DTE — Direct Task Engagement:** task activity on the laptop.
- **TREE — Task-Relevant External Engagement:** off-screen activity that is still relevant to the task.
- **TIE — Task-Irrelevant Engagement:** activity that is not relevant to the assigned task.

For binary classification, DTE and TREE are mapped to **on-task**, while TIE is mapped to **distraction**.

## Study artifact

The final analyzed cohort contained:

- **18 participants** from four Indonesian universities;
- **30 s** individual calibration per session;
- **6 × 120 s** counterbalanced blocks per participant;
- **12,015** quality-screened overlapping windows;
- **2 s** windows with **1 s** stride;
- resampling at **5 Hz**;
- **21 final model features** derived from head orientation, gaze proxy, validity indicators, and gaze-category representation.

The final paper dataset excludes body features because they were not sufficiently observed in the analyzed recordings. Roll was also excluded cohort-wide because calibration was unstable for two participants.

## Main reported results

| Task | Method | Context | Evaluation | Macro-F1 |
|---|---|---:|---:|---:|
| Binary | Random Forest | 2 s | 5 folds | 0.721 ± 0.044 |
| Binary | Tuned GRU | 2 s | 5 folds | 0.738 ± 0.021 |
| Binary | Validation-selected neural models | 6 s | 5 folds | **0.747 ± 0.047** |
| Three-class | Neural | 6 s | 6 folds | 0.630 ± 0.020 |
| Three-class | Regularized tabular + causal EMA | 2 s | 6 folds | **0.647 ± 0.064** |

For the best binary experiment, pooled distraction recall was **0.600**. Correct classification reached **93.8% for DTE** and **83.6% for TREE**. These results are exploratory and should not be interpreted as deployment-ready performance.

See [results/PAPER_RESULTS.md](results/PAPER_RESULTS.md).

## Repository contents

```text
backend/          Local research API and runtime
frontend/         Participant/operator/stimulus interfaces
preprocessing/    Raw-data audit and preprocessing
training/         RF, neural, tabular, ablation, and evaluation pipelines
configs/          Research and runtime configuration
tests/            Automated tests
docs/             Protocol, methodology, data dictionary, reproducibility
models/           Released fold-specific trained model artifacts
results/          Aggregate and fold-level evaluation artifacts
data/             Data availability statement (no participant data)
```

## Data availability and privacy

**Participant-level research data are not publicly released in this repository.** This includes raw webcam recordings, consent forms, identity linkage, participant responses, and the processed participant-level feature dataset.

The consent documentation used for the study did not explicitly authorize unrestricted public redistribution of participant-level research data. Therefore this repository contains code, configuration, documentation, aggregate/fold-level metrics, and selected trained model artifacts only.

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

## Trained models

The directory `models/binary_5fold_6s/` contains the five fold-specific neural models selected by validation for the best reported binary experiment. The selected architecture can differ between folds; the reported score represents a model-selection procedure rather than one universal architecture.

See [models/README.md](models/README.md).

## Setup

Recommended environment: Python 3.11 or 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

To rebuild the frontend:

```powershell
cd frontend
npm ci
npm run build
cd ..
```

## Run the local research application

```powershell
python run.py
```

Engineering-only synthetic mode:

```powershell
python run.py --dev --synthetic-camera
```

Synthetic outputs must not be treated as participant evidence.

## Reproducibility

The repository includes the preprocessing, grouped participant splitting, model training, and evaluation code used during the research workflow. Because participant-level data are not public, an exact end-to-end numerical reproduction from the original cohort is not possible from this public repository alone.

For pipeline verification without human data:

```powershell
python scripts\generate_demo_data.py --output outputs\synthetic\dataset.npz --participants 18 --seconds-per-block 8
```

Then run the relevant training commands described in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Scientific limitations

- The study includes 18 participants; thousands of overlapping windows do not equal thousands of independent people.
- Most labels originate from assigned experimental activities; only a subset underwent manual behavioral review.
- Webcam gaze is a proxy rather than validated eye-tracker coordinates.
- Performance varies across participants and task forms.
- The strongest error pattern remains the distinction between relevant and irrelevant external activities.
- Independent participants, comprehensive annotation, and naturalistic evaluation are required before educational deployment.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff).

## Availability statement for the manuscript

> Source code, preprocessing and training pipelines, experiment configurations, selected trained model artifacts, and reproducibility materials for FocusTrace are publicly available in this repository. Raw webcam recordings and participant-level research data are not publicly released to protect participant privacy and because the study consent documentation did not explicitly authorize unrestricted public redistribution.
