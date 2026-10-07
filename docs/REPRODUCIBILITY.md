# Reproducibility Guide

## What can be reproduced publicly

The public repository supports:

- installation and application startup;
- feature-extraction code inspection;
- synthetic dataset generation;
- participant-grouped splitting logic;
- model training and evaluation on compatible datasets;
- inspection of released fold-level metrics and selected binary model artifacts.

Exact numerical reproduction of the paper from the original cohort is intentionally not possible from the public repository because participant-level research data are not redistributed.

## Environment

Use Python 3.11 or 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-training.txt
python -m pip install -e .
```

## Synthetic pipeline smoke test

```powershell
python scripts\generate_demo_data.py --output outputs\synthetic\dataset.npz --participants 18 --seconds-per-block 8
```

Then use the training scripts with participant grouping. Synthetic outputs are for engineering verification only and must not be compared with the paper's participant results as scientific evidence.

## Real-data preprocessing specification

The final paper dataset used:

- 18 included participants;
- 2 s windows;
- 1 s stride;
- 5 Hz temporal grid;
- maximum sample age 0.3 s;
- gaps greater than 0.5 s break a segment;
- minimum 80% valid head-or-gaze grid points;
- no windows crossing participant, session, block, or label boundaries;
- calibration-relative features computed without task labels;
- participant-separated train/validation/test roles;
- imputers and normalization fitted only on training participants.

See `configs/research_18.json` for the frozen research configuration.

## Binary best experiment

The best complete binary procedure used five participant-separated folds and six seconds of preceding/current context. Candidate neural families included GRU, LSTM, TCN, and temporal-summary MLP. Configuration and decision threshold were selected using validation participants inside each fold.

The released `models/binary_5fold_6s/fold_*.pt` files correspond to the selected model in each fold.

## Three-class best experiment

The best mean three-class result used regularized tabular models with causal probability smoothing (EMA) and six participant-separated folds. EMA state was reset at session/block boundaries and long gaps. The public repository includes the implementation, but not participant-level prediction traces.
