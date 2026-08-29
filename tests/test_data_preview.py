import utils.data_preview as data_preview


def test_missing_transient_log_is_reported_unavailable(tmp_path):
    """A deleted candidate log must not abort the advisory data preview."""
    missing_log = tmp_path / "download.log"

    assert data_preview.get_file_len_size(missing_log) == (0, "unavailable")


def test_preview_skips_log_deleted_after_directory_walk(tmp_path, monkeypatch):
    transient_log = tmp_path / "download.log"
    transient_log.write_text("temporary\n")
    monkeypatch.setattr(data_preview, "file_tree", lambda _path: "")

    def delete_after_measurement(path):
        path.unlink()
        return 1, "1 lines"

    monkeypatch.setattr(data_preview, "get_file_len_size", delete_after_measurement)

    assert data_preview.generate(tmp_path) == "```\n```"
