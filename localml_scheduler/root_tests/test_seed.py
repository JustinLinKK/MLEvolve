import sys
from types import SimpleNamespace

from utils.seed import set_global_seed


def test_global_seed_seeds_torch_and_all_cuda_devices(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("manual_seed", seed)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("manual_seed_all", seed)),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    set_global_seed(42)

    assert calls == [("manual_seed", 42), ("manual_seed_all", 42)]
