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


def test_resume_script_resets_only_stale_watchdog_observations():
    script = (
        Path(__file__).parents[1]
        / "deployments"
        / "resume_petfinder_scheduler_on_a10_boot.sh"
    ).read_text()

    assert 'rm -f "$watch_dir/last_count"' in script
    assert '"$watch_dir/last_progress_epoch"' in script
    assert '"$watch_dir/STALL_DETECTED.json"' in script
