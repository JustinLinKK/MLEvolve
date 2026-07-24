# Stress Test Data v1.0

## Required Kaggle dataset

This stress test represents the
[Kaggle Histopathologic Cancer Detection competition](https://www.kaggle.com/competitions/histopathologic-cancer-detection),
a binary classification task for identifying metastatic cancer in
histopathology image patches.

Use the official competition download above. Do not substitute a similarly
named Kaggle community dataset, because its files or preprocessing may differ.
You must join the competition, accept its rules, and configure Kaggle API
credentials before the CLI download will work.

```bash
export HISTOPATH_DATA_ROOT=/absolute/path/to/histopathologic-cancer-detection
mkdir -p "$HISTOPATH_DATA_ROOT"
kaggle competitions download \
  --competition histopathologic-cancer-detection \
  --path "$HISTOPATH_DATA_ROOT"
unzip "$HISTOPATH_DATA_ROOT/histopathologic-cancer-detection.zip" \
  -d "$HISTOPATH_DATA_ROOT"
unzip "$HISTOPATH_DATA_ROOT/train.zip" -d "$HISTOPATH_DATA_ROOT/train"
unzip "$HISTOPATH_DATA_ROOT/test.zip" -d "$HISTOPATH_DATA_ROOT/test"
```

After extraction, the important files should be:

```text
$HISTOPATH_DATA_ROOT/
├── train/
│   └── <image-id>.tif
├── test/
│   └── <image-id>.tif
├── train_labels.csv
└── sample_submission.csv
```

The raw Kaggle dataset must remain outside this repository and must not be
committed.

## What this repository contains

This directory contains the versioned 100-job model list and model-source
fixture, not the Kaggle images:

- `joblist.json`: 100 one-epoch scheduler job specifications;
- `model_source.py`: the compatible model structures;
- `manifest.json`: dataset version, distributions, schema dimensions, and
  deterministic hashes.

The CPU predictor compatibility verifier does not read the Kaggle images. The
official dataset is needed when these model structures are used for the real
Histopathologic Cancer Detection training stress test.

Verify all 100 model structures with:

```bash
python -m scheduler_benchmark_test.standard.stress_test_data \
  --check --verify-predictions
```
