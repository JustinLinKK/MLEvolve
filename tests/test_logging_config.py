from __future__ import annotations

import logging
from types import SimpleNamespace

from utils.logging_config import setup_logging


def test_setup_logging_replaces_handlers_and_disables_propagation(tmp_path) -> None:
    cfg = SimpleNamespace(log_dir=tmp_path, log_level="INFO")

    logger = setup_logging(cfg)
    logger = setup_logging(cfg)

    assert logger is logging.getLogger("MLEvolve")
    assert logger.propagate is False
    assert len(logger.handlers) == 3

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
