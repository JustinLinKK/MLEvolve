from agents.training_contract_validation import validate_training_contract


def test_neural_training_requires_quality_envelope_early_stop_and_epoch_markers() -> None:
    code = """
import torch
BATCH_SIZE = 16
EPOCHS = 5
for epoch in range(EPOCHS):
    loss.backward()
    optimizer.step()
"""

    categories = {issue.category for issue in validate_training_contract(code)}

    assert categories == {
        "batch_quality_envelope",
        "batch_optimizer_coupling",
        "validation_early_stopping",
        "epoch_progress_reporting",
    }


def test_complete_neural_training_contract_passes() -> None:
    code = """
import torch
import json
BATCH_SIZE = 16
EPOCHS = 5
QUALITY_SAFE_PHYSICAL_BATCH_SIZES = [8, 16, 32]
BATCH_LR_SCALING_POLICY = "fixed"
patience = 2
for epoch in range(EPOCHS):
    loss.backward()
    optimizer.step()
    validation_score = validate()
    print('MLEVOLVE_EPOCH_METRIC ' + json.dumps({'epoch': epoch + 1, 'metric': validation_score, 'metric_name': 'validation_score'}))
    if no_improve >= patience:
        break
"""

    assert validate_training_contract(code) == ()


def test_identifier_column_must_not_be_compared_with_positional_split_indices() -> None:
    code = """
import numpy as np
import pandas as pd

def make_val_split(frame):
    idx = np.arange(len(frame))
    val_idx = set(idx[:2].tolist())
    train_mask = ~frame['Id'].isin(list(val_idx))
    return frame[train_mask], frame[~train_mask]
"""

    issues = validate_training_contract(code)

    assert {issue.category for issue in issues} == {"identifier_index_split"}
