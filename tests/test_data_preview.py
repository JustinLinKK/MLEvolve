from utils.data_preview import get_file_len_size


def test_missing_transient_log_is_reported_unavailable(tmp_path):
    """A deleted candidate log must not abort the advisory data preview."""
    missing_log = tmp_path / "download.log"

    assert get_file_len_size(missing_log) == (0, "unavailable")
