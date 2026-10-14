"""Forward-looking full experiments budget generated nodes, not successes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_experiment_defaults_are_fifty_nodes():
    expected = {
        "Agent/ML-Master/utils/config_mcts.yaml": "  steps: 50",
        "localml_scheduler/config.example.yaml": "  steps: 50",
        "localml_scheduler/deployments/run_petfinder_aba_a100_phase.sh":
            'node_budget="${SCHEDULER_NODE_BUDGET:-50}"',
        "localml_scheduler/deployments/run_petfinder_aba_a100_preflight_only.sh":
            'node_budget="${PREFLIGHT_NODE_BUDGET:-50}"',
        "localml_scheduler/deployments/run_petfinder_seeded_baseline_a10.sh":
            'node_budget="${PETFINDER_NODE_BUDGET:-50}"',
    }
    for path, setting in expected.items():
        assert setting in (ROOT / path).read_text(encoding="utf-8")
    preflight_launcher = (
        ROOT / "localml_scheduler/deployments/run_petfinder_aba_a100_preflight_only.sh"
    ).read_text(encoding="utf-8")
    assert 'test "$node_budget" -eq 50' in preflight_launcher


def test_runner_progress_counts_persisted_buggy_nodes():
    for path in ("Agent/ML-Master/main_mcts.py", "Agent/MLEvolve/run.py"):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "completed=len(journal)-1" in source.replace(" ", "")
