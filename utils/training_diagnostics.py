"""Low-frequency runtime evidence for generated PyTorch training scripts."""

from __future__ import annotations

import json
import time
import uuid


TRAINING_DIAGNOSTICS_MARKER = "MLEVOLVE_TRAINING_DIAGNOSTICS"
TRAINING_DIAGNOSTICS_INSTRUCTION = (
    "PyTorch runtime diagnostics: import TrainingDiagnostics from utils.training_diagnostics. "
    "Inside the training entrypoint, wrap training in `with TrainingDiagnostics(model, optimizer, "
    "scaler=scaler_or_None, settings={...}) as diagnostics:` after model/optimizer initialization. "
    "Settings must include actual physical_batch_size, effective_batch_size, planned_epochs, "
    "steps_per_epoch, model_family, split_seed and precision choice/reason. "
    "Call diagnostics.after_update() once after each optimizer update ATTEMPT, after scaler.step "
    "and scaler.update (or optimizer.step); never after accumulation microbatches. "
    "The helper observes real optimizer steps, including skips, and actual autocast dtype. "
    "Call diagnostics.report(epoch=one_based_epoch) once per epoch; keep the normal epoch metric line. "
    "Save diagnostics.state_dict() with the training checkpoint and restore it using "
    "diagnostics.load_state_dict(...). Preserve these calls during merges/repairs. "
    "Do not replace missing measurements with zero. Keep the final validation score as the last line, "
    "after leaving the diagnostics context. These diagnostics also run without the scheduler."
)


class TrainingDiagnostics:
    """Count optimizer calls without changing updates or synchronizing each step.

    Use one instance per optimizer/training stream. ``after_update`` counts an
    attempt; the optimizer's public post-step hook counts successful calls, so a
    GradScaler overflow is observable even when the loss itself was finite.
    """

    def __init__(self, model, optimizer, *, scaler=None, settings=None):
        import torch

        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.settings = dict(settings or {})
        self.session_id = uuid.uuid4().hex
        self.started_at = time.time()
        self.attempted_updates = 0
        self.completed_updates = 0
        self.skipped_updates = 0
        self._pending_completed = 0
        self._fused_overflows = []
        self._autocast = set()
        self.device = next(model.parameters()).device
        self.seed = torch.initial_seed()
        self.torch_version = torch.__version__
        self.gpu = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None
        self._optimizer_hook = optimizer.register_step_post_hook(self._after_optimizer_step)
        self._forward_hook = model.register_forward_pre_hook(self._before_forward)

    def _after_optimizer_step(self, optimizer, args, kwargs):
        self._pending_completed += 1
        # Fused AMP optimizers may enter step() and skip inside the CUDA kernel.
        # Read their overflow flags at reporting/checkpoint boundaries only.
        found_inf = getattr(optimizer, "found_inf", None)
        if found_inf is not None:
            self._fused_overflows.append(found_inf.detach().clone())

    def _before_forward(self, model, args):
        import torch

        if model.training:
            enabled = torch.is_autocast_enabled(self.device.type)
            dtype = str(torch.get_autocast_dtype(self.device.type)) if enabled else "disabled"
            self._autocast.add(dtype)

    def after_update(self):
        self.attempted_updates += 1
        self.completed_updates += self._pending_completed
        self.skipped_updates += int(self._pending_completed == 0)
        self._pending_completed = 0

    def state_dict(self):
        if self._fused_overflows:
            import torch

            skipped = int(torch.stack(self._fused_overflows).ne(0).sum().item())
            self.completed_updates -= skipped
            self.skipped_updates += skipped
            self._fused_overflows.clear()
        return {
            "attempted_updates": self.attempted_updates,
            "completed_updates": self.completed_updates,
            "skipped_updates": self.skipped_updates,
        }

    def load_state_dict(self, state):
        for key in ("attempted_updates", "completed_updates", "skipped_updates"):
            setattr(self, key, int(state[key]))

    def report(self, *, epoch=None, status="running", exception_type=None):
        import torch

        parameter_dtypes = sorted({str(p.dtype) for p in self.model.parameters()})
        state_dtypes = sorted({
            str(value.dtype)
            for state in self.optimizer.state.values()
            for value in state.values()
            if hasattr(value, "dtype") and value.is_floating_point()
        })
        payload = {
            "schema_version": 1,
            "session_id": self.session_id,
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - self.started_at,
            "status": status,
            "exception_type": exception_type,
            "epoch": epoch,
            "settings": self.settings,
            "gpu": self.gpu,
            "device": str(self.device),
            "torch_version": self.torch_version,
            "torch_seed": self.seed,
            "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn_allowed": torch.backends.cudnn.allow_tf32,
            "autocast_dtypes_observed": sorted(self._autocast),
            "parameter_dtypes": parameter_dtypes,
            "optimizer_state_dtypes": state_dtypes,
            "learning_rates": [float(group["lr"]) for group in self.optimizer.param_groups],
            "grad_scaler_enabled": bool(self.scaler is not None and self.scaler.is_enabled()),
            "grad_scale": self.scaler.get_scale() if self.scaler is not None else None,
            "unaccounted_optimizer_calls": self._pending_completed,
            **self.state_dict(),
        }
        print(TRAINING_DIAGNOSTICS_MARKER + " " + json.dumps(payload, allow_nan=False, default=str), flush=True)
        return payload

    def __enter__(self):
        self.report(status="initialized")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            interrupted = exc_type is not None and exc_type.__name__ in {
                "PauseRequested", "CancelRequested", "KeyboardInterrupt", "SystemExit",
            }
            self.report(status="interrupted" if interrupted else "failed" if exc_type else "completed",
                        exception_type=exc_type.__name__ if exc_type else None)
        finally:
            self._optimizer_hook.remove()
            self._forward_hook.remove()
