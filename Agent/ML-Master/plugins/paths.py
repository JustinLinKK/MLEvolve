"""Locate Lightning Agent's shared plugins and upstream agents."""

from pathlib import Path
import sys


MLMASTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MLMASTER_ROOT.parents[1]
MLEVOLVE_ROOT = MLMASTER_ROOT.parent / "MLEvolve"


def expose_components() -> Path:
    """Make root-level scheduler and checker packages importable."""
    if not PROJECT_ROOT.is_dir():
        raise FileNotFoundError(f"Lightning Agent root is missing: {PROJECT_ROOT}")
    for path in (PROJECT_ROOT,):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return PROJECT_ROOT
