from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from localml_scheduler.adapters.mlevolve_runner import (
    _base_script_env,
    _infer_planned_epochs,
    _record_live_epoch_metric,
    probe_mlevolve_script_job,
    run_mlevolve_script_job,
)
from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.checkpointing.manager import CheckpointManager
from localml_scheduler.execution.control import ControlPlane, TrainingControlHook
from localml_scheduler.execution.runner_protocol import RunnerContext
from localml_scheduler.observability.events import EventLogger
from localml_scheduler.domain import (
    BatchProbeSpec,
    BatchResolution,
    CheckpointPolicy,
    PackingSpec,
    RuntimeProbeSpec,
    SafePointType,
    TrainingJob,
)
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def _build_context(settings: SchedulerSettings, job: TrainingJob) -> RunnerContext:
    store = SQLiteStateStore(settings)
    store.save_job(job)
    event_logger = EventLogger(store, settings.events_jsonl_path)
    checkpoint_manager = CheckpointManager(settings, store, event_logger)
    control_plane = ControlPlane(settings)
    control_plane.initialize_job(job.job_id)
    control_hook = TrainingControlHook(
        job, control_plane, checkpoint_manager, store, event_logger
    )
    return RunnerContext(
        job=job,
        settings=settings,
        store=store,
        event_logger=event_logger,
        control_hook=control_hook,
        checkpoint_manager=checkpoint_manager,
        cache_client=None,
    )


class MLEvolveRunnerTest(unittest.TestCase):
    def test_mlevolve_job_propagates_configured_seed_to_subprocess_runner(self) -> None:
        job = build_mlevolve_job(
            workflow_id="seeded-workflow",
            baseline_model_id="seeded-script",
            baseline_model_path="/tmp/candidate.py",
            runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
            seed=42,
        )

        self.assertEqual(job.config.seed, 42)
        self.assertFalse(job.metadata["supports_checkpoint_resume"])

    def test_instrumented_raw_script_sets_python_numpy_and_torch_seed(self) -> None:
        """A subprocess must not rely on the controller process's random state."""

        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir()
            script_path = working_dir / "candidate.py"
            result_path = working_dir / "result.json"
            script_path.write_text(
                "import json\n"
                "import random\n"
                "import numpy as np\n"
                "import torch\n"
                "print(json.dumps([random.random(), float(np.random.rand()), float(torch.rand(1))]))\n"
                "print('determinism', torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)\n",
                encoding="utf-8",
            )

            settings = SchedulerSettings(runtime_root=Path(tmpdir) / "runtime")
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="seeded-raw-script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(result_path),
                    "timeout": 30,
                },
                packing=PackingSpec(signature="seeded-raw-script"),
                seed=42,
            )
            context = _build_context(settings, job)
            run_mlevolve_script_job(context)
            execution = json.loads(result_path.read_text(encoding="utf-8"))
            values_line = next(
                line for line in execution["term_out"] if line.lstrip().startswith("[")
            )
            python_value, numpy_value, torch_value = json.loads(values_line)

            self.assertAlmostEqual(python_value, 0.6394267984578837)
            self.assertAlmostEqual(numpy_value, 0.3745401188473625)
            self.assertAlmostEqual(torch_value, 0.8822692632675171)
            self.assertTrue(
                any(
                    "determinism True True False" in line
                    for line in execution["term_out"]
                )
            )

    def test_runtime_env_preserves_candidate_loader_worker_setting_by_default(self) -> None:
        """The runner must not silently turn a GPU input pipeline into serial CPU loading."""

        self.assertNotIn("MLEVOLVE_FORCE_NUM_WORKERS", _base_script_env())

    def test_runner_infers_typed_dataclass_epoch_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "candidate.py"
            script_path.write_text(
                "from dataclasses import dataclass\n"
                "@dataclass\n"
                "class TrainConfig:\n"
                "    epochs: int = 45\n",
                encoding="utf-8",
            )

            self.assertEqual(_infer_planned_epochs(script_path), 45)

    def test_runner_infers_argparse_epoch_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "candidate.py"
            script_path.write_text(
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--epochs', type=int, default=40)\n"
                "args = parser.parse_args()\n"
                "for epoch in range(1, args.epochs + 1):\n"
                "    pass\n",
                encoding="utf-8",
            )

            self.assertEqual(_infer_planned_epochs(script_path), 40)

    def test_live_epoch_metric_keeps_store_canonical_epoch_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=Path(tmpdir) / "runtime")
            job = TrainingJob.create(
                runner_target="pkg.runner:train",
                baseline_model_id="stale-epoch-context",
                baseline_model_path="/tmp/candidate.py",
                runtime_probe=RuntimeProbeSpec(enabled=True),
                packing=PackingSpec(signature="stale-epoch-context"),
                metadata={"placement_backend": "exclusive", "planned_epochs": 4},
            )
            context = _build_context(settings, job)
            persisted = context.store.get_job(job.job_id)
            assert persisted is not None
            persisted.max_epochs = 45
            persisted.config.max_epochs = 45
            persisted.metadata["planned_epochs"] = 45
            context.store.save_job(persisted)

            _record_live_epoch_metric(
                context,
                "Epoch 6: train_loss=0.4 valid_rmse=0.3",
                started_at=time.time() - 12.0,
            )

            live = context.store.get_job(job.job_id)
            assert live is not None
            self.assertEqual(live.max_epochs, 45)
            self.assertEqual(live.config.max_epochs, 45)
            self.assertEqual(live.metadata["planned_epochs"], 45)
            self.assertEqual(live.metadata["last_completed_epoch"], 6)

    def test_live_epoch_metric_recalibrates_runtime_from_later_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=Path(tmpdir) / "runtime")
            job = TrainingJob.create(
                runner_target="pkg.runner:train",
                baseline_model_id="runtime-recalibration",
                baseline_model_path="/tmp/candidate.py",
                max_epochs=45,
                runtime_probe=RuntimeProbeSpec(enabled=True),
                packing=PackingSpec(signature="runtime-recalibration"),
                metadata={"placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            _record_live_epoch_metric(
                context,
                "Epoch 1: train_loss=0.4 valid_rmse=0.3",
                started_at=time.time() - 2.0,
            )
            _record_live_epoch_metric(
                context,
                "Epoch 6: train_loss=0.3 valid_rmse=0.2",
                started_at=time.time() - 120.0,
            )

            profile = context.get_runtime_profile(backend_name="exclusive")
            assert profile is not None
            self.assertGreater(float(profile.estimated_total_runtime_seconds or 0.0), 800.0)

    def test_live_epoch_metric_uses_monotonic_count_across_named_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=Path(tmpdir) / "runtime")
            job = TrainingJob.create(
                runner_target="pkg.runner:train",
                baseline_model_id="two-stage-script",
                baseline_model_path="/tmp/candidate.py",
                max_epochs=20,
                runtime_probe=RuntimeProbeSpec(enabled=True),
                packing=PackingSpec(signature="two-stage-script"),
                metadata={"placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            _record_live_epoch_metric(
                context,
                "Stage1 Epoch 5/5 | Val RMSE: 0.4",
                started_at=time.time() - 50.0,
            )
            _record_live_epoch_metric(
                context,
                "Stage2 Epoch 1/15 | Val RMSE: 0.3",
                started_at=time.time() - 60.0,
            )

            persisted = context.store.get_job(job.job_id)
            assert persisted is not None
            self.assertEqual(persisted.metadata["last_completed_epoch"], 6)
            self.assertEqual(persisted.metadata["planned_epochs"], 20)
            self.assertEqual(persisted.metadata["runtime_stage_epoch_offset"], 5)

    def test_live_epoch_metric_records_epoch_intervals_for_colocation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=Path(tmpdir) / "runtime")
            job = TrainingJob.create(
                runner_target="pkg.runner:train",
                baseline_model_id="colocation-live-epochs",
                baseline_model_path="/tmp/candidate.py",
                runtime_probe=RuntimeProbeSpec(enabled=True),
                packing=PackingSpec(signature="colocation-live-epochs"),
                metadata={"placement_backend": "cuda_process"},
            )
            context = _build_context(settings, job)

            _record_live_epoch_metric(
                context,
                "Epoch 1: valid_rmse=0.4",
                started_at=time.time() - 2.0,
            )
            _record_live_epoch_metric(
                context,
                "Epoch 2: valid_rmse=0.3",
                started_at=time.time() - 4.0,
            )

            persisted = context.store.get_job(job.job_id)
            assert persisted is not None
            history = persisted.metadata["runtime_epoch_timing_history"]
            self.assertEqual([sample["epoch"] for sample in history], [1, 2])
            self.assertGreater(history[1]["seconds"], 0.0)
            self.assertEqual(history[1]["started_at"], history[0]["finished_at"])
            self.assertIsNotNone(history[1]["finished_at"])

    def test_runtime_profile_is_persisted_before_script_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import time",
                        "print('MLEVOLVE_EPOCH_METRIC {\"epoch\": 1, \"metric\": 0.7, \"metric_name\": \"validation_score\"}', flush=True)",
                        "time.sleep(1.0)",
                        "print('MLEVOLVE_EPOCH_METRIC {\"epoch\": 2, \"metric\": 0.8, \"metric_name\": \"validation_score\"}', flush=True)",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="streaming-script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(working_dir / "result.json"),
                    "timeout": 30,
                },
                runtime_probe=RuntimeProbeSpec(enabled=True),
                max_epochs=2,
                packing=PackingSpec(signature="streaming-runtime-profile"),
                metadata={"placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)
            result: dict[str, object] = {}
            worker = threading.Thread(
                target=lambda: result.update(run_mlevolve_script_job(context))
            )
            worker.start()
            deadline = time.monotonic() + 0.8
            profiles = []
            while time.monotonic() < deadline:
                profiles = context.store.list_runtime_profiles(
                    signature="streaming-runtime-profile"
                )
                if profiles:
                    break
                time.sleep(0.02)
            self.assertTrue(worker.is_alive())
            self.assertEqual(len(profiles), 1)
            self.assertIsNotNone(profiles[0].estimated_total_runtime_seconds)
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result["candidate_returncode"], 0)
            calibrated = context.store.list_runtime_profiles(
                signature="streaming-runtime-profile"
            )
            self.assertGreaterEqual(
                float(calibrated[0].estimated_total_runtime_seconds or 0.0), 0.9
            )

    def test_standard_epoch_output_creates_live_runtime_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import time",
                        "epochs = 4",
                        "print('Epoch 1: train_loss=0.7 valid_rmse=0.5', flush=True)",
                        "time.sleep(1.0)",
                        "print('Epoch 2: train_loss=0.6 valid_rmse=0.4', flush=True)",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="standard-epoch-script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(working_dir / "result.json"),
                    "timeout": 30,
                },
                runtime_probe=RuntimeProbeSpec(enabled=True),
                packing=PackingSpec(signature="standard-epoch-runtime-profile"),
                metadata={
                    "placement_backend": "exclusive",
                    "planned_epochs": 4,
                },
            )
            context = _build_context(settings, job)
            result: dict[str, object] = {}
            worker = threading.Thread(
                target=lambda: result.update(run_mlevolve_script_job(context))
            )
            worker.start()
            deadline = time.monotonic() + 0.8
            profiles = []
            while time.monotonic() < deadline:
                profiles = context.store.list_runtime_profiles(
                    signature="standard-epoch-runtime-profile"
                )
                live = context.store.get_job(job.job_id)
                if profiles and live is not None and "last_completed_epoch" in live.metadata:
                    break
                time.sleep(0.02)
            self.assertTrue(worker.is_alive())
            self.assertEqual(len(profiles), 1)
            self.assertIsNotNone(profiles[0].estimated_total_runtime_seconds)
            live = context.store.get_job(job.job_id)
            self.assertEqual(live.metadata["last_completed_epoch"], 1)
            self.assertEqual(live.metadata["planned_epochs"], 4)
            self.assertEqual(live.max_epochs, 4)
            self.assertEqual(live.config.max_epochs, 4)
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result["candidate_returncode"], 0)

    def test_run_script_resolves_coupled_training_parameters_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import json",
                        "from pathlib import Path",
                        "from torch.utils.data import DataLoader",
                        "batch_size = 4",
                        "gradient_accumulation_steps = 4",
                        "learning_rate = 0.01",
                        "warmup_steps = 10",
                        "total_training_steps = 100",
                        "train_loader = DataLoader(list(range(20)), batch_size=batch_size)",
                        "Path('resolved.json').write_text(json.dumps({",
                        "    'batch': train_loader.batch_size,",
                        "    'accumulation': gradient_accumulation_steps,",
                        "    'learning_rate': learning_rate,",
                        "    'warmup_steps': warmup_steps,",
                        "    'scheduler_total_steps': total_training_steps,",
                        "}), encoding='utf-8')",
                        "print('MLEVOLVE_EPOCH_METRIC {\"epoch\": 1, \"metric\": 0.7, \"metric_name\": \"validation_score\"}')",
                        "print('MLEVOLVE_EPOCH_METRIC {\"epoch\": 2, \"metric\": 0.8, \"metric_name\": \"validation_score\"}')",
                        "print('Best epoch: 2')",
                        "print('Final Validation Score: 0.8')",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(working_dir / "result.json"),
                    "timeout": 30,
                    "batch_size": 4,
                },
                batch_probe=BatchProbeSpec(model_key="quality-script"),
                max_epochs=3,
                metadata={
                    "placement_backend": "exclusive",
                    "training_quality_contract": {
                        "proposed_physical_batch_size": 4,
                        "allowed_physical_batch_sizes": [4, 8],
                        "base_gradient_accumulation_steps": 4,
                        "target_effective_batch_size": 16,
                        "base_learning_rate": 0.01,
                        "learning_rate_scaling_policy": "linear",
                        "base_warmup_steps": 10,
                        "base_scheduler_total_steps": 100,
                        "planned_epochs": 3,
                    },
                },
            )
            job = BatchResolution.apply(job, 8)
            context = _build_context(settings, job)

            result = run_mlevolve_script_job(context)

            self.assertEqual(result["candidate_returncode"], 0)
            resolved = json.loads(
                (working_dir / "resolved.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                resolved,
                {
                    "batch": 8,
                    "accumulation": 2,
                    "learning_rate": 0.01,
                    "warmup_steps": 10,
                    "scheduler_total_steps": 100,
                },
            )
            observations = context.store.list_batch_size_observations(
                model_key="quality-script"
            )
            self.assertEqual(len(observations), 1)
            observation = observations[0]
            self.assertEqual(observation.batch_size, 8)
            self.assertEqual(observation.effective_batch_size, 16)
            self.assertEqual(observation.planned_epochs, 3)
            self.assertEqual(observation.completed_epochs, 2)
            self.assertEqual(observation.best_epoch, 2)
            self.assertEqual(observation.best_metric, 0.8)
            self.assertEqual(len(observation.convergence_curve), 2)

    def test_run_script_job_honors_resolved_batch_size_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        "from torch.utils.data import DataLoader",
                        "batch_size = 4",
                        "train_loader = DataLoader(list(range(20)), batch_size=batch_size, shuffle=True)",
                        "Path('batch_size.txt').write_text(str(train_loader.batch_size), encoding='utf-8')",
                    ]
                ),
                encoding="utf-8",
            )
            result_path = working_dir / "result.json"
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(result_path),
                    "timeout": 30,
                },
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
                ),
                checkpoint_policy=CheckpointPolicy(
                    save_every_n_steps=1, pause_mode=SafePointType.STEP
                ),
                metadata={"resolved_batch_size": 9, "placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            result = run_mlevolve_script_job(context)

            self.assertEqual(result["candidate_returncode"], 0)
            self.assertEqual(result["batch_size_override"], 9)
            self.assertEqual(
                (working_dir / "batch_size.txt").read_text(encoding="utf-8"), "9"
            )

    def test_probe_script_job_runs_successfully_with_batch_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import time",
                        "batch_size = 2",
                        "time.sleep(1)",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(working_dir / "result.json"),
                    "timeout": 30,
                    "probe_timeout_seconds": 1,
                    "probe_poll_interval_seconds": 0.2,
                },
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
                ),
                checkpoint_policy=CheckpointPolicy(
                    save_every_n_steps=1, pause_mode=SafePointType.STEP
                ),
                metadata={"placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            result = probe_mlevolve_script_job(
                context, batch_size=5, warmup_steps=1, measure_steps=1
            )

            self.assertTrue(result.fits)
            self.assertIn("probe", result.message or "")

    def test_probe_script_job_limits_epochs_and_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import json",
                        "from pathlib import Path",
                        "from torch.utils.data import DataLoader",
                        "batch_size = 2",
                        "num_epochs = 5",
                        "trace = []",
                        "train_loader = DataLoader(list(range(20)), batch_size=batch_size)",
                        "for epoch in range(num_epochs):",
                        "    for step, batch in enumerate(train_loader):",
                        "        trace.append({'epoch': epoch, 'step': step, 'size': int(len(batch))})",
                        "Path('probe_trace.json').write_text(json.dumps(trace), encoding='utf-8')",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(working_dir / "result.json"),
                    "timeout": 30,
                    "probe_timeout_seconds": 5,
                    "probe_poll_interval_seconds": 0.2,
                    "probe_max_epochs": 1,
                    "probe_max_train_batches": 3,
                },
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
                ),
                checkpoint_policy=CheckpointPolicy(
                    save_every_n_steps=1, pause_mode=SafePointType.STEP
                ),
                metadata={"placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            result = probe_mlevolve_script_job(
                context, batch_size=7, warmup_steps=1, measure_steps=1
            )

            self.assertTrue(result.fits)
            trace = json.loads(
                (working_dir / "probe_trace.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(trace), 3)
            self.assertEqual({entry["epoch"] for entry in trace}, {0})
            self.assertEqual([entry["step"] for entry in trace], [0, 1, 2])
            self.assertEqual([entry["size"] for entry in trace], [7, 7, 6])


if __name__ == "__main__":
    unittest.main()
