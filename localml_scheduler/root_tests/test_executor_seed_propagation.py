from types import SimpleNamespace

from engine.executor import Interpreter


def _baseline_cfg(seed: int):
    return SimpleNamespace(
        agent=SimpleNamespace(
            seed=seed,
            search=SimpleNamespace(parallel_search_num=None),
        ),
        scheduler=SimpleNamespace(enabled=False),
        start_cpu_id=0,
        cpu_number=1,
    )


def test_baseline_subprocess_injects_configured_code_seed(tmp_path) -> None:
    """The unscheduled baseline must seed candidate code just like a scheduler worker."""
    interpreter = Interpreter(tmp_path, timeout=30, max_parallel_run=1, cfg=_baseline_cfg(42))
    code = (
        "import random\n"
        "import numpy as np\n"
        "import torch\n"
        "print(random.random(), np.random.rand(), torch.rand(1).item())\n"
        "print('determinism', torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)\n"
    )

    first = interpreter.run(code, id="first")
    second = interpreter.run(code, id="second")

    assert first.exc_type is None
    assert second.exc_type is None
    first_value = next(line for line in first.term_out if line.strip() and not line.startswith("Execution time"))
    second_value = next(line for line in second.term_out if line.strip() and not line.startswith("Execution time"))
    assert first_value == second_value
    assert any("determinism True True False" in line for line in first.term_out)
    assert any("determinism True True False" in line for line in second.term_out)
