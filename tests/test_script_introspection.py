from pathlib import Path

from engine.script_introspection import analyze_training_batch_contract, introspect_training_script
from localml_scheduler.adapters.mlevolve_runner import _materialize_instrumented_script


def test_training_batch_contract_detects_alias_and_minimum() -> None:
    code = """
BASE_BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BASE_BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BASE_BATCH_SIZE * 2)
while BASE_BATCH_SIZE >= 8:
    break
"""

    contract = analyze_training_batch_contract(code)

    assert contract.supported
    assert contract.initial_batch_size == 64
    assert contract.minimum_batch_size == 8
    assert introspect_training_script(code)["proposed_batch_size"] == 64


def test_introspection_populates_exact_and_broad_architecture_identity() -> None:
    cnn = introspect_training_script(
        'MODEL_NAME = "resnet50"\nBATCH_SIZE = 4\ntrain_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)\n'
    )
    transformer = introspect_training_script(
        'MODEL_NAME = "vit_b16"\nBATCH_SIZE = 4\ntrain_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)\n'
    )
    cnn_with_transformer_engine = introspect_training_script(
        'MODEL_NAME = "resnet50"\nimport transformer_engine.pytorch as te\n'
    )

    assert cnn["architecture_key"] == "resnet50"
    assert cnn["architecture_family"] == "cnn"
    assert transformer["architecture_key"] == "vit-b16"
    assert transformer["architecture_family"] == "transformer"
    assert cnn_with_transformer_engine["architecture_family"] == "cnn"


def test_training_batch_contract_resolves_function_parameter_fallbacks() -> None:
    code = """
PHYSICAL_BATCH_SIZE = 48
FALLBACK_BATCH_SIZE = 24
def train(batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
try:
    train(PHYSICAL_BATCH_SIZE)
except RuntimeError:
    train(FALLBACK_BATCH_SIZE)
"""

    contract = analyze_training_batch_contract(code)

    assert contract.supported
    assert contract.initial_batch_size == 48
    assert set(contract.batch_symbols) == {"FALLBACK_BATCH_SIZE", "PHYSICAL_BATCH_SIZE", "batch_size"}


def test_training_batch_contract_resolves_dict_config_subscript() -> None:
    code = """
from torch.utils.data import DataLoader

TRAINING_CONFIG = {"physical_batch_size": 96, "eval_batch_size": 192}
train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG["physical_batch_size"], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=TRAINING_CONFIG["eval_batch_size"], shuffle=False)
"""

    contract = analyze_training_batch_contract(code)

    assert contract.supported
    assert contract.initial_batch_size == 96
    assert contract.train_sites[0].expression == "TRAINING_CONFIG['physical_batch_size']"


def test_training_batch_contract_resolves_config_object_attribute() -> None:
    code = """
from dataclasses import dataclass
from torch.utils.data import DataLoader

@dataclass
class TrainConfig:
    physical_batch_size: int = 80

TRAIN_CONFIG = TrainConfig()
train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG.physical_batch_size, shuffle=True)
"""

    contract = analyze_training_batch_contract(code)

    assert contract.supported
    assert contract.initial_batch_size == 80


def test_training_batch_contract_resolves_caller_attribute_argument() -> None:
    code = """
from types import SimpleNamespace
from torch.utils.data import DataLoader

config = SimpleNamespace(batch_size=72)

def train(batch_size):
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return loader

train(config.batch_size)
"""

    contract = analyze_training_batch_contract(code)

    assert contract.supported
    assert contract.initial_batch_size == 72


def test_training_batch_contract_resolves_instrumented_caller_argument() -> None:
    code = """
from torch.utils.data import DataLoader

_MLEVOLVE_BATCH_SIZE_OVERRIDE = None
BATCH_SIZE = 128

def get_dataloaders(batch_size, num_workers):
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(_MLEVOLVE_BATCH_SIZE_OVERRIDE) if _MLEVOLVE_BATCH_SIZE_OVERRIDE is not None else batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(_MLEVOLVE_BATCH_SIZE_OVERRIDE) if _MLEVOLVE_BATCH_SIZE_OVERRIDE is not None else batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, valid_loader

get_dataloaders(
    batch_size=int(_MLEVOLVE_BATCH_SIZE_OVERRIDE) if _MLEVOLVE_BATCH_SIZE_OVERRIDE is not None else BATCH_SIZE,
    num_workers=4,
)
"""

    contract = analyze_training_batch_contract(code)

    assert contract.supported
    assert contract.initial_batch_size == 128
    assert contract.train_sites[0].lineno == 8


def test_batch_instrumentation_is_scoped_to_training_loader(tmp_path: Path) -> None:
    script = tmp_path / "candidate.py"
    script.write_text(
        """
from torch.utils.data import DataLoader
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
for labels in train_loader:
    observed_batch_size = labels.size(0)
""",
        encoding="utf-8",
    )

    result = _materialize_instrumented_script(script, tmp_path)
    materialized = result.path.read_text(encoding="utf-8")

    assert result.had_batch_rewrite
    assert materialized.count("_MLEVOLVE_BATCH_SIZE_OVERRIDE is not None") == 1
    assert "valid_loader = DataLoader(valid_dataset, batch_size=batch_size)" in materialized
    assert "test_loader = DataLoader(test_dataset, batch_size=batch_size)" in materialized
    assert "observed_batch_size = labels.size(0)" in materialized


def test_ambiguous_batch_variable_is_not_probeable() -> None:
    contract = analyze_training_batch_contract("batch_size = 32\nprint(batch_size)\n")

    assert not contract.supported
    assert "training DataLoader" in str(contract.unsupported_reason)


def test_loader_factory_contract_tracks_only_proven_train_call() -> None:
    code = """
from torch.utils.data import DataLoader

INITIAL_PHYSICAL_BATCH_SIZE = 32
MIN_PHYSICAL_BATCH_SIZE = 4

def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def fit(train_df, valid_df, batch_size):
    train_loader = make_loader(train_df, batch_size, shuffle=True)
    valid_loader = make_loader(valid_df, batch_size, shuffle=False)
    return train_loader, valid_loader

batch_size = INITIAL_PHYSICAL_BATCH_SIZE
while batch_size >= MIN_PHYSICAL_BATCH_SIZE:
    fit(train_df, valid_df, batch_size)
    batch_size //= 2
"""

    contract = analyze_training_batch_contract(code)

    assert contract.supported
    assert contract.initial_batch_size == 32
    assert contract.minimum_batch_size == 4
    assert [(site.argument, site.expression) for site in contract.train_sites] == [
        ("positional:1", "batch_size")
    ]
