# Released model artifacts

`binary_5fold_6s/` contains the five fold-specific neural model files selected by validation in the best reported binary experiment.

Important points:

1. These are **research evaluation artifacts**, not one production model.
2. Architecture selection differed by fold; the manuscript reports the performance of the validation-based selection procedure.
3. Each model was trained using participant-separated train/validation/test roles.
4. The original participant-level preprocessing state is not released publicly, so these files should not be treated as a turnkey deployment package.
5. Do not use these artifacts to make disciplinary, grading, diagnostic, or high-stakes decisions about students.

Reported mean macro-F1 across the five folds: **0.747 ± 0.047**.
