# Results reported in the manuscript

This file summarizes the final values reported in the accompanying FocusTrace manuscript.

## Dataset

- Participants: 18
- Windows: 12,015
- DTE: 3,934
- TREE: 4,084
- TIE: 3,997
- On-task (DTE + TREE): 8,018
- Distraction (TIE): 3,997
- Window length: 2 s
- Stride: 1 s
- Grid: 5 Hz
- Final model features: 21

## Binary classification — five folds

| Method | Context | Macro-F1 ± SD | Accuracy |
|---|---:|---:|---:|
| Always on-task | — | 0.401 ± 0.009 | 0.670 |
| Random Forest baseline | 2 s | 0.721 ± 0.044 | 0.771 |
| Tuned GRU | 2 s | 0.738 ± 0.021 | 0.777 |
| Neural selection | 2 s | 0.724 ± 0.035 | 0.765 |
| Neural selection | 6 s | **0.747 ± 0.047** | **0.790** |

Pooled best-binary evaluation:

- On-task precision: 0.816
- On-task recall: 0.886
- On-task F1: 0.850
- Distraction precision: 0.724
- Distraction recall: 0.600
- Distraction F1: 0.656
- Pooled accuracy: 0.791
- Pooled macro-F1: 0.753

Condition-level proportion classified correctly in the best binary experiment:

- DTE: 93.80%
- TREE: 83.64%
- TIE: 59.97%

## Three-class classification — six folds

| Method | Context | Macro-F1 ± SD | Accuracy |
|---|---:|---:|---:|
| Tabular movement | 2 s | 0.616 ± 0.042 | 0.621 |
| Tabular movement | 6 s | 0.624 ± 0.060 | 0.627 |
| Neural | 6 s | 0.630 ± 0.020 | 0.632 |
| Regularized tabular | 2 s | 0.608 ± 0.044 | 0.612 |
| Regularized tabular + causal EMA | 2 s | **0.647 ± 0.064** | **0.651** |
| Regularized tabular + causal EMA | 6 s | 0.645 ± 0.072 | 0.647 |

For the best three-class procedure:

- DTE recall: 0.806; F1: 0.763
- TREE recall: 0.561; F1: 0.552
- TIE recall: 0.594; F1: 0.642

## Interpretation boundary

These values describe exploratory classification under the study protocol. They do not demonstrate direct measurement of psychological focus, comprehension, or readiness for real-world educational deployment.
