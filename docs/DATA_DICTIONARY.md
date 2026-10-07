# Data Dictionary

This document distinguishes **collection-time fields** from the **final 21-feature paper representation**.

## Core identifiers and metadata

Identifiers such as participant code, session ID, timestamp, block code, condition, and labels are used for synchronization, grouping, traceability, and evaluation. They are **not model input features** in the final paper experiments.

## Collection-time visual signals

The runtime may record:

- head pose estimates (yaw, pitch, roll);
- gaze horizontal/vertical proxy and category;
- validity/detection indicators;
- optional face/pose/body-motion signals;
- calibration-relative values.

These collection-time columns are broader than the final model feature set.

## Final paper model features: 21

### Head — 8
Sine/cosine representations of absolute and calibration-relative yaw/pitch.

### Gaze — 6
Horizontal, vertical, gaze deviation, duration outside center, and calibration-relative horizontal/vertical values.

### Validity — 2
Head-valid and gaze-valid indicators.

### Gaze category — 5
One-hot representation of CENTER, LEFT, RIGHT, DOWN, and UNKNOWN.

**Body features are not part of the final paper dataset. Roll is also excluded from the final paper model inputs.**

## Window metadata

The processed research dataset also retains non-model metadata needed for audit and grouped evaluation, including participant group, block, window start time, source provenance, and label information.

## Public-data restriction

Participant-level feature rows, responses, event logs, raw video, and session databases are not distributed in this public repository. See `DATA_AVAILABILITY.md`.
