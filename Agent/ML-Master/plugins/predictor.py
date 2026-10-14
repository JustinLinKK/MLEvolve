"""Opt-in PerfSeer predictor adapter for ML-Master candidate source."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT


logger = logging.getLogger("ml-master")


class PredictorPlugin:
    """Predict only when an entry class and input shapes are explicitly known."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.student_dir = PROJECT_ROOT / "PerfSeer-predictor" / "student"
        self.artifact = Path(
            os.environ.get(
                "LIGHTNING_PREDICTOR_ARTIFACT",
                str(self.student_dir / "student_A10.pt"),
            )
        )
        self._model: Any = None

    def predict_source(
        self,
        source: Path,
        *,
        entry: str,
        input_shapes: list[list[int]],
        precision: str = "fp32_ieee",
    ) -> dict[str, float]:
        if not self.artifact.is_file():
            raise FileNotFoundError(f"PerfSeer artifact missing: {self.artifact}")
        if self._model is None:
            spec = importlib.util.spec_from_file_location(
                "lightning_perfseer_predict", self.student_dir / "predict.py"
            )
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load PerfSeer student predictor")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._model = module.SeerPredictor(self.artifact)
        return self._model.predict_source(
            str(source), entry, input_shapes, precision
        )

    def inspect_candidate(self, code: str, node_id: str) -> dict[str, float] | None:
        entry = os.environ.get("LIGHTNING_PREDICTOR_ENTRY")
        shapes_json = os.environ.get("LIGHTNING_PREDICTOR_INPUT_SHAPES")
        if not entry or not shapes_json:
            logger.warning(
                "Predictor enabled but entry/input shapes absent; set "
                "LIGHTNING_PREDICTOR_ENTRY and LIGHTNING_PREDICTOR_INPUT_SHAPES"
            )
            return None
        report_dir = self.workspace / "working" / "predictor"
        report_dir.mkdir(parents=True, exist_ok=True)
        source = report_dir / f"candidate_{node_id}.py"
        source.write_text(code, encoding="utf-8")
        try:
            shapes = json.loads(shapes_json)
            if not isinstance(shapes, list) or not all(
                isinstance(shape, list) and all(isinstance(n, int) and n > 0 for n in shape)
                for shape in shapes
            ):
                raise ValueError("input shapes must be a JSON array of positive integer arrays")
            result = self.predict_source(
                source,
                entry=entry,
                input_shapes=shapes,
                precision=os.environ.get("LIGHTNING_PREDICTOR_PRECISION", "fp32_ieee"),
            )
        except Exception as exc:
            logger.warning("PerfSeer prediction unavailable for node %s: %s", node_id, exc)
            (report_dir / f"result_{node_id}.json").write_text(
                json.dumps({"error": f"{type(exc).__name__}: {exc}"}), encoding="utf-8"
            )
            return None
        (report_dir / f"result_{node_id}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result
