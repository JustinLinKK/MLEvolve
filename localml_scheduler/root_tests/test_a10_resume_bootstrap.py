from pathlib import Path


def test_resume_script_falls_back_to_a_persistent_python_environment():
    script = (
        Path(__file__).parents[1]
        / "deployments"
        / "resume_petfinder_scheduler_on_a10_boot.sh"
    ).read_text()

    assert "python_fallback=" in script
    assert "if [[ ! -x \"$python_bin\" ]]" in script
    assert 'python_bin="$python_fallback"' in script


def test_resume_script_keeps_disposable_watchdog_state_off_the_persistent_volume():
    script = (
        Path(__file__).parents[1]
        / "deployments"
        / "resume_petfinder_scheduler_on_a10_boot.sh"
    ).read_text()

    assert "watch_dir=/dev/shm/mlevolve_scheduler_watchdog_a10_v7" in script
    assert 'rm -f "$watch_dir/last_count"' not in script


def test_resume_script_exposes_the_checked_out_preflight_package():
    script = (
        Path(__file__).parents[1]
        / "deployments"
        / "resume_petfinder_scheduler_on_a10_boot.sh"
    ).read_text()

    assert 'PYTHONPATH="$repo/nn-model-preflight-checker/src' in script
