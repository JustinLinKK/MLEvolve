from __future__ import annotations

from utils.candidate_timing import instrument_code_for_phase_timing


def test_phase_instrumentation_preserves_future_import_preamble(tmp_path) -> None:
    code = '''"""candidate module"""
from __future__ import annotations

def train_model():
    return 1
'''

    result = instrument_code_for_phase_timing(code, tmp_path / "phases.jsonl")

    assert result.instrumented is True
    compile(result.code, "candidate.py", "exec")
    assert result.code.index("from __future__ import annotations") < result.code.index(
        "import json as _mlevolve_phase_json"
    )
