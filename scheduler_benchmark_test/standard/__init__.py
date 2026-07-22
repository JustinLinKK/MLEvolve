"""Standard, deterministic histopathology scheduler benchmark."""

from pathlib import Path


SCHEMA_VERSION = "standard-histopath-v1"
FIXTURE_NAME = "standard_histopath_v1"
JOB_COUNT = 100
EPOCHS = 50
DATASET_SIZE = 174_464
INPUT_SIZE = 96
INITIAL_BATCH_SIZE = 32
MAX_PROBE_BATCH_SIZE = 256
ARRIVAL_RATE = 0.1
SEED = 42
A10_VRAM_CAP_MIB = 22_528

PACKAGE_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = PACKAGE_ROOT.parent
DEFAULT_FIXTURE_ROOT = BENCHMARK_ROOT / "fixtures" / FIXTURE_NAME

