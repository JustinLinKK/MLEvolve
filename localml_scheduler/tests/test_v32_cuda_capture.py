from __future__ import annotations

import os
from multiprocessing import Pipe
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from localml_scheduler.prediction.v32_source_worker import (
    _capture_request_with_small_batch,
    _training_settings,
)


HAS_V32_CUDA_FIXTURES = bool(
    torch.cuda.is_available()
    and os.environ.get("PERFSEER_V32_SOURCE_ROOT")
    and os.environ.get("PERFSEER_V32_ARTIFACT")
)


class V32CudaCaptureTest(unittest.TestCase):
    def test_capture_training_settings_preserve_steps_per_epoch(self) -> None:
        settings = _training_settings(
            {
                "input_shapes": [[32, 3, 8, 8]],
                "precision": "fp32",
                "steps_per_epoch": 279,
            }
        )

        self.assertEqual(settings["steps_per_epoch"], 279)

    def test_source_capture_uses_a_small_graph_batch_but_keeps_real_batch_metadata(self) -> None:
        """CPU graph capture must not allocate a full training microbatch."""
        request = {
            "input_shapes": [[32, 3, 384, 384], [32, 12]],
            "precision": "fp32",
            "steps_per_epoch": 279,
        }

        captured = _capture_request_with_small_batch(request)

        self.assertEqual(captured["input_shapes"], [[2, 3, 384, 384], [2, 12]])
        self.assertEqual(_training_settings(request)["microbatch_size"], 32)
        self.assertEqual(_training_settings(captured)["microbatch_size"], 32)

    @unittest.skipUnless(HAS_V32_CUDA_FIXTURES, "requires CUDA and Predictor v3.2 fixtures")
    def test_strict_capture_replays_dropout_with_cuda_rng_state(self) -> None:
        source_root = Path(os.environ["PERFSEER_V32_SOURCE_ROOT"]).resolve()
        sys.path.insert(0, str(source_root))
        from perfseer_v31.capture_training import capture

        class DropoutModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(8, 8)
                self.dropout = torch.nn.Dropout(p=0.5)

            def forward(self, values: torch.Tensor) -> torch.Tensor:
                return self.dropout(self.projection(values))

        model = DropoutModel().cuda().train()
        inputs = (torch.ones((4, 8), device="cuda"),)
        design, audit = capture(
            model,
            inputs,
            None,
            None,
            {
                "microbatch_size": 4,
                "gradient_accumulation_steps": 1,
                "precision": "fp32",
                "backend": "cuda_eager",
                "optimizer": {"name": "adam", "learning_rate": 1e-3, "weight_decay": 0.0},
                "scheduler": {"name": "none"},
                "steps_per_epoch": 1,
            },
            architecture_key="dropout-cuda-rng-regression",
        )
        self.assertEqual(design["version"], "perfseer_v31_model_design_v1")
        self.assertGreater(audit["gradient_parameters_checked"], 0)

    @unittest.skipUnless(HAS_V32_CUDA_FIXTURES, "requires CUDA and Predictor v3.2 fixtures")
    def test_v32_worker_returns_predictions_from_cuda_capture(self) -> None:
        source_root = Path(os.environ["PERFSEER_V32_SOURCE_ROOT"]).resolve()
        artifact_path = Path(os.environ["PERFSEER_V32_ARTIFACT"]).resolve()
        if not artifact_path.is_file():
            self.skipTest("requires Predictor v3.2 artifact")
        from localml_scheduler.prediction.v32_source_worker import (
            predict_v32_source_options_worker,
        )

        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            candidate.write_text(
                "import torch.nn as nn\n"
                "def build_model():\n"
                "    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 1))\n",
                encoding="utf-8",
            )
            receiver, sender = Pipe(duplex=False)
            predict_v32_source_options_worker(
                sender,
                str(source_root),
                str(artifact_path),
                [
                    {
                        "source_path": str(candidate),
                        "entry": "build_model",
                        "input_shapes": [[4, 3, 8, 8]],
                        "input_dtypes": ["float32"],
                        "constructor_args": [],
                        "constructor_kwargs": {},
                        "precision": "fp32",
                    }
                ],
                capture_device="cuda",
            )
            response = receiver.recv()
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["capture_device"], "cuda")
        prediction = response["predictions"][0]
        self.assertGreater(prediction["train_epoch_ms"], 0.0)
        self.assertGreater(prediction["train_peak_vram_mib"], 0.0)
