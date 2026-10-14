# PetFinder.my - Pawpularity Score

Predict the continuous `Pawpularity` target for the test pet profiles. Use only
the supplied competition data: `input/train.csv`, `input/test.csv`, and the
corresponding JPEGs in `input/train/` and `input/test/`. The identifier column
is `Id`; training labels are in `Pawpularity`.

Train and validate a GPU-capable model with a reproducible validation split.
The selection metric is root mean squared error (RMSE), so lower is better.
Report the final validation RMSE in stdout as `Final Validation Score: <value>`.
Write predictions for every test row, in original test-row order, to
`submission/submission.csv` with exactly the columns `Id,Pawpularity`.

Do not use external data, network access, or target leakage. Keep the run
practical for a single NVIDIA A100 80GB GPU.
