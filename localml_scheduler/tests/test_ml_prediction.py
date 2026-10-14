from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from localml_scheduler.config import PredictionSettings, SchedulerSettings
from localml_scheduler.domain import ResourceRequirements, TrainingJob
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.prediction import (
    JobPredictionError,
    MLVramPredictor,
    model_specification_for_job,
)
from localml_scheduler.prediction.ml_predictor import (
    JobPredictionPending,
    ModelSpecification,
    _PendingV32Capture,
)
from localml_scheduler.prediction.ml_adapter import PerfSeerMLAdapter
from localml_scheduler.scheduler.resource_estimator import ResourceEstimator


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "PerfSeer-predictor" / "models" / "nvidia_a10" / "student_a10_cpu.torchscript.pt"
BLACKWELL_ARTIFACT = (
    ROOT
    / "PerfSeer-predictor"
    / "models"
    / "nvidia_rtx_pro_6000_blackwell"
    / "student_rtx_pro_6000_blackwell_cpu.torchscript.pt"
)
V32_SOURCE_ROOT = (
    ROOT.parent / "Predictor" / "predictor_v3.2" / "source"
)
V32_ARTIFACT = (
    ROOT.parent
    / "Predictor"
    / "predictor_v3.2"
    / "weights"
    / "predictor_v3.2_transfer_A100.pt"
)
SUPPORTED_SOURCE = Path(__file__).parent / "fixtures" / "tiny_conv.py"
UNSUPPORTED_SOURCE = Path(__file__).parent / "fixtures" / "perfseer_unsupported.py"
TIMEOUT_SOURCE = Path(__file__).parent / "fixtures" / "perfseer_timeout.py"


def hardware(name: str, capability: str, memory_mb: int) -> HardwareProfile:
    return HardwareProfile("test-hardware", "linux", name, memory_mb, capability, "13.0", "test")


def model_job(source: Path, *, job_id: str = "predict-job") -> TrainingJob:
    return TrainingJob.create(
        "localml_scheduler.examples.toy_pytorch_runner:run_toy_job",
        "tiny",
        str(source),
        job_id=job_id,
        resource_requirements=ResourceRequirements(estimated_avg_vram_mb=777),
        metadata={
            "perfseer_model": {
                "source_path": str(source),
                "entry": "build_model",
                "input_shapes": [["$batch", 3, 32, 32]],
                "input_dtypes": ["float32"],
                "precision": "fp32",
            }
        },
    )


class _Repository:
    def __init__(self, profile: HardwareProfile):
        self.profile = profile

    def hardware_profile(self):
        return self.profile

    def hardware_key(self):
        return self.profile.hardware_key

    def get_batch_size_observation(self, **kwargs):
        return None

    def list_batch_size_observations(self, **kwargs):
        return []

    def get_batch_probe_profile(self, key):
        return None

    def get_solo_profile(self, signature):
        return None


class _IncompatibleOverride(torch.nn.Module):
    def forward(self, x, edge_index, edge_attr, u, batch):
        return x.new_zeros((1, 1))


class MLPredictionTest(unittest.TestCase):
    def test_v32_async_capture_reports_a_dead_worker_before_deadline(self) -> None:
        """A crashed capture cannot masquerade as pending until its timeout."""

        class _Receiver:
            closed = False

            def poll(self, timeout: float) -> bool:
                return False

            def close(self) -> None:
                self.closed = True

        class _Process:
            exitcode = 1

            def is_alive(self) -> bool:
                return False

            def join(self, timeout: float = 0.0) -> None:
                return None

        predictor = object.__new__(MLVramPredictor)
        predictor._v32_pending_capture = _PendingV32Capture(
            receiver=_Receiver(),
            process=_Process(),
            keys=("failed-key",),
            deadline=10**12,
        )

        with self.assertRaisesRegex(JobPredictionError, "exited without a result"):
            predictor._poll_v32_capture()
        self.assertIsNone(predictor._v32_pending_capture)

    def test_v32_async_capture_returns_pending_then_caches_completed_result(self) -> None:
        """A slow source capture must not block the scheduler service tick."""

        class _Receiver:
            def __init__(self) -> None:
                self.ready = False
                self.closed = False

            def poll(self, timeout: float) -> bool:
                self.last_timeout = timeout
                return self.ready

            def recv(self):
                return {
                    "ok": True,
                    "predictions": [{"train_peak_vram_mib": 512.0, "train_epoch_ms": 250.0}],
                }

            def close(self) -> None:
                self.closed = True

        class _Sender:
            def close(self) -> None:
                return None

        class _Process:
            def __init__(self, **kwargs) -> None:
                self.started = False
                self.terminated = False
                self.kwargs = kwargs

            def start(self) -> None:
                self.started = True

            def is_alive(self) -> bool:
                return False

            def join(self, timeout: float = 0.0) -> None:
                return None

            def terminate(self) -> None:
                self.terminated = True

        class _Context:
            def __init__(self, receiver: _Receiver) -> None:
                self.receiver = receiver
                self.process = _Process()

            def Pipe(self, duplex: bool = False):
                return self.receiver, _Sender()

            def Process(self, **kwargs):
                self.process.kwargs = kwargs
                return self.process

        predictor = object.__new__(MLVramPredictor)
        predictor.settings = PredictionSettings(
            mode="ml_predictor",
            v32_source_root="/tmp/source",
            v32_artifact_path="/tmp/artifact.pt",
            v32_async_source_capture=True,
        )
        predictor._v32_predictions = {}
        predictor._v32_pending_capture = None
        receiver = _Receiver()
        context = _Context(receiver)
        specifications = [
            ModelSpecification(
                source_path=SUPPORTED_SOURCE,
                working_directory=SUPPORTED_SOURCE.parent,
                entry="build_model",
                input_shapes=((batch_size, 3, 32, 32),),
                input_dtypes=("float32",),
                precision="fp32_ieee",
                constructor_args=(),
                constructor_kwargs={},
            )
            for batch_size in (8, 16, 32)
        ]
        job = model_job(SUPPORTED_SOURCE)
        job.config.runner_kwargs["batch_size"] = 32

        with patch("localml_scheduler.prediction.ml_predictor.multiprocessing.get_context", return_value=context):
            with self.assertRaises(JobPredictionPending):
                predictor._predict_v32_options(job, specifications, ["key-8", "key-16", "key-32"])
            self.assertTrue(context.process.started)
            self.assertFalse(hasattr(receiver, "last_timeout"))
            capture_requests = context.process.kwargs["args"][3]
            self.assertEqual(len(capture_requests), 1)
            self.assertEqual(capture_requests[0]["input_shapes"][0][0], 32)

            receiver.ready = True
            predictor._predict_v32_options(job, specifications, ["key-8", "key-16", "key-32"])

        self.assertEqual(predictor._v32_predictions["key-32"]["train_peak_vram_mib"], 512.0)

    def test_v32_worker_request_scales_epoch_by_dataset_cardinality(self) -> None:
        job = model_job(SUPPORTED_SOURCE)
        job.metadata["perfseer_model"]["dataset_size"] = 8920

        request = model_specification_for_job(job, 32).worker_request()

        self.assertEqual(request["steps_per_epoch"], 279)

    def test_v32_worker_request_keeps_the_execution_directory_for_relative_inputs(self) -> None:
        job = model_job(SUPPORTED_SOURCE)
        execution_directory = SUPPORTED_SOURCE.parent
        job.config.runner_kwargs["working_dir"] = str(execution_directory)

        request = model_specification_for_job(job, 32).worker_request()

        self.assertEqual(request["working_directory"], str(execution_directory))

    def test_v32_capture_device_round_trips_through_scheduler_settings(self) -> None:
        settings = PredictionSettings(
            mode="ml_predictor",
            v32_source_root="/tmp/source",
            v32_artifact_path="/tmp/artifact.pt",
            v32_capture_device="cuda",
        )
        self.assertEqual(settings.v32_capture_device, "cuda")
        self.assertEqual(
            PredictionSettings.from_dict(settings.to_dict()).v32_capture_device,
            "cuda",
        )

    @unittest.skipUnless(
        V32_SOURCE_ROOT.is_dir() and V32_ARTIFACT.is_file(),
        "local PerfSeer v3.2 package is unavailable",
    )
    def test_v32_a10_checkpoint_predicts_peak_vram_and_epoch_time(self) -> None:
        predictor = MLVramPredictor(
            PredictionSettings(
                mode="ml_predictor",
                v32_source_root=str(V32_SOURCE_ROOT),
                v32_artifact_path=str(V32_ARTIFACT),
                conversion_timeout_seconds=60,
            ),
            hardware("NVIDIA A10", "8.6", 23028),
        )
        self.assertTrue(predictor.available, predictor.unavailable_reason)
        job = model_job(SUPPORTED_SOURCE)
        self.assertGreater(predictor.predict_avg_vram_mb(job, 2), 0)
        self.assertGreater(predictor.predict_seconds_per_epoch_options(job, [2])[2], 0)
        self.assertEqual(predictor.last_sources["predict-job"], "ml_predictor")

    def test_auto_selects_a10_and_runs_cpu_prediction(self) -> None:
        predictor = MLVramPredictor(
            PredictionSettings(mode="ml_predictor"),
            hardware("NVIDIA A10", "8.6", 23028),
        )
        self.assertTrue(predictor.available, predictor.unavailable_reason)
        value = predictor.predict_avg_vram_mb(model_job(SUPPORTED_SOURCE), 2)
        self.assertGreater(value, 0)
        self.assertEqual(predictor.last_sources["predict-job"], "ml_predictor")

    def test_five_batch_predictions_share_one_conversion_bundle_and_cache(self) -> None:
        predictor = MLVramPredictor(
            PredictionSettings(mode="ml_predictor"),
            hardware("NVIDIA A10", "8.6", 23028),
        )
        self.assertTrue(predictor.available, predictor.unavailable_reason)
        calls: list[int] = []
        original = predictor._convert_many

        def counted(specifications):
            calls.append(len(specifications))
            return original(specifications)

        predictor._convert_many = counted  # type: ignore[method-assign]
        values = predictor.predict_avg_vram_options(model_job(SUPPORTED_SOURCE), [1, 2, 4, 8, 16])
        self.assertEqual(set(values), {1, 2, 4, 8, 16})
        self.assertTrue(all(value > 0 for value in values.values()))
        self.assertEqual(calls, [5])
        cached = predictor.predict_avg_vram_options(model_job(SUPPORTED_SOURCE), [1, 2, 4, 8, 16])
        self.assertEqual(cached, values)
        self.assertEqual(calls, [5])

    def test_auto_selects_rtx_pro_6000_blackwell_and_runs_cpu_prediction(self) -> None:
        predictor = MLVramPredictor(
            PredictionSettings(mode="ml_predictor"),
            hardware(
                "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "12.0",
                98304,
            ),
        )
        self.assertTrue(predictor.available, predictor.unavailable_reason)
        self.assertEqual(
            predictor.model_id,
            "nvidia_rtx_pro_6000_blackwell_student_v1",
        )
        self.assertEqual(predictor._runtime.artifact_path, BLACKWELL_ARTIFACT)
        value = predictor.predict_avg_vram_mb(model_job(SUPPORTED_SOURCE), 2)
        self.assertGreater(value, 0)
        self.assertEqual(predictor.last_sources["predict-job"], "ml_predictor")

    def test_rejects_non_a10_without_override(self) -> None:
        predictor = MLVramPredictor(
            PredictionSettings(mode="ml_predictor"),
            hardware("NVIDIA GeForce RTX 5090", "12.0", 32607),
        )
        self.assertFalse(predictor.available)
        with self.assertRaises(JobPredictionError):
            predictor.predict_avg_vram_mb(model_job(SUPPORTED_SOURCE), 2)

    def test_explicit_test_override_runs_on_other_hardware(self) -> None:
        for artifact in (ARTIFACT, BLACKWELL_ARTIFACT):
            with self.subTest(artifact=artifact.name):
                predictor = MLVramPredictor(
                    PredictionSettings(
                        mode="ml_predictor",
                        test_override_enabled=True,
                        test_model_path=str(artifact),
                    ),
                    hardware("NVIDIA GeForce RTX 5090", "12.0", 32607),
                )
                self.assertTrue(predictor.available, predictor.unavailable_reason)
                self.assertGreater(
                    predictor.predict_avg_vram_mb(model_job(SUPPORTED_SOURCE), 2),
                    0,
                )

    def test_override_rejects_incompatible_torchscript_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "incompatible.torchscript.pt"
            example = (
                torch.zeros((2, 53)),
                torch.tensor([[0], [1]], dtype=torch.long),
                torch.zeros((1, 3)),
                torch.zeros((1, 40)),
                torch.zeros(2, dtype=torch.long),
            )
            torch.jit.trace(_IncompatibleOverride(), example).save(str(path))
            predictor = MLVramPredictor(
                PredictionSettings(
                    mode="ml_predictor",
                    test_override_enabled=True,
                    test_model_path=str(path),
                ),
                hardware("NVIDIA GeForce RTX 5090", "12.0", 32607),
            )
            self.assertFalse(predictor.available)
            self.assertIn("shape", predictor.unavailable_reason)

    def test_unsupported_job_falls_back_without_disabling_ml(self) -> None:
        settings = SchedulerSettings(
            runtime_root=Path(tempfile.gettempdir()) / "unused-predictor-test",
            prediction={
                "mode": "ml_predictor",
                "ml": {
                    "test_override_enabled": True,
                    "test_model_path": str(ARTIFACT),
                },
            },
        )
        estimator = ResourceEstimator(settings, _Repository(hardware("RTX 5090", "12.0", 32607)))
        unsupported = model_job(UNSUPPORTED_SOURCE, job_id="unsupported")
        supported = model_job(SUPPORTED_SOURCE, job_id="supported")
        self.assertEqual(estimator.estimate_avg_vram_mb(unsupported, 2, "exclusive"), 1554)
        self.assertEqual(unsupported.metadata["vram_prediction_source"], "branch_profile")
        self.assertGreater(estimator.estimate_avg_vram_mb(supported, 2, "exclusive"), 0)
        self.assertEqual(supported.metadata["vram_prediction_source"], "ml_predictor")

    def test_records_model_specification_failure_for_required_ml_admission(self) -> None:
        """A missing input contract must not become a silent branch-profile rejection."""
        predictor = object.__new__(MLVramPredictor)
        predictor.available = True
        predictor.last_sources = {}
        predictor.last_errors = {}
        predictor._job_failures = {}
        malformed = model_job(SUPPORTED_SOURCE, job_id="missing-input-contract")
        del malformed.metadata["perfseer_model"]["input_shapes"]

        with self.assertRaisesRegex(JobPredictionError, "input_shapes"):
            predictor.predict_avg_vram_options(malformed, [2])

        self.assertEqual(
            predictor.last_sources["missing-input-contract"], "branch_profile"
        )
        self.assertIn(
            "input_shapes", predictor.last_errors["missing-input-contract"]
        )

    def test_conversion_timeout_is_per_job_failure(self) -> None:
        predictor = MLVramPredictor(
            PredictionSettings(
                mode="ml_predictor",
                conversion_timeout_seconds=0.1,
                test_override_enabled=True,
                test_model_path=str(ARTIFACT),
            ),
            hardware("RTX 5090", "12.0", 32607),
        )
        with self.assertRaisesRegex(JobPredictionError, "exceeded"):
            predictor.predict_avg_vram_mb(model_job(TIMEOUT_SOURCE), 2)


if __name__ == "__main__":
    unittest.main()
