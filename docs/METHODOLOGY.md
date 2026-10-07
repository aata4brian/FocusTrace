# Methodology Notes

FocusTrace analyzes **observable task-related behavior** under controlled experimental conditions. Labels are defined by task relevance and limited behavioral review; gaze direction alone is never equated with distraction.

## Final paper feature set

The collection software can record a broader set of visual features, including pose/body indicators. However, the final analyzed paper dataset uses **21 model features** based on head orientation and gaze because body features were not sufficiently observed in the analyzed recordings.

The final feature groups are:

- **Head (8):** sine/cosine representation of absolute and calibration-relative yaw/pitch.
- **Gaze (6):** horizontal, vertical, deviation, duration outside center, and calibration-relative horizontal/vertical values.
- **Validity (2):** head-valid and gaze-valid indicators.
- **Category (5):** one-hot gaze category: CENTER, LEFT, RIGHT, DOWN, UNKNOWN.

Roll was excluded cohort-wide before modeling because calibration was unstable for two participants.

## Personalized calibration

A 30-second pre-block calibration estimates participant-specific head and gaze centers. Relative features are then computed against that baseline. Calibration for test participants does not use task labels.

## Temporal dataset

Rows are placed on a **5 Hz** temporal grid and grouped into **2-second windows** with **1-second stride**. A gap greater than 0.5 s breaks a segment. A window must contain sufficient valid visual observations and may not cross participant, session, block, or label boundaries.

Six-second causal context is formed only from current and preceding windows; no future samples are used.

## Labels

Three-class labels:

- DTE — Direct Task Engagement
- TREE — Task-Relevant External Engagement
- TIE — Task-Irrelevant Engagement

Binary mapping:

- DTE + TREE → on-task
- TIE → distraction

Most labels follow the assigned experimental activity. A limited subset of 185 windows uses behavioral review corrections. The dataset is therefore not described as comprehensively frame-annotated.

## Evaluation

All reported evaluation is participant-separated. Train, validation, and test participants are disjoint inside each fold. Imputation, normalization, model selection, and threshold selection follow the corresponding training/validation roles to reduce participant leakage.

The primary binary comparison uses five folds. Three-class exploratory analysis uses six folds.

## Interpretation

Outputs must not be described as direct measurements of psychological focus, concentration, motivation, or comprehension. The strongest unresolved error pattern is distinguishing task-relevant external activity from task-irrelevant external activity.
