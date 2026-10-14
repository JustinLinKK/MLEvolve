"""Evidence-first lesson extraction and structured LLM summarization."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Mapping

from .config import LessonProfileSettings
from .models import LessonRecord
from .registry import LessonProfileRegistry


SummaryGenerator = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]

_CAUSAL_TERMS = re.compile(r"\b(?:caused|causes|because of|resulted in|led to|improved due to)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")


def _confidence(success_runs: int, conflict_count: int) -> float:
    value = min(0.95, 0.35 + 0.2 * max(1, success_runs))
    return max(0.1, value - 0.15 * max(0, conflict_count))


def _metric_value(evidence: Mapping[str, Any]) -> float | None:
    value = (evidence.get("validation") or {}).get("metric")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _known_numeric_tokens(evidence: Mapping[str, Any]) -> set[str]:
    serialized = json.dumps(evidence, sort_keys=True, default=str)
    return set(_NUMBER.findall(serialized))


class LessonBuilder:
    """Build a revision without treating model prose as measured evidence."""

    def __init__(
        self,
        settings: LessonProfileSettings,
        registry: LessonProfileRegistry,
        *,
        cfg: Any | None = None,
        summary_generator: SummaryGenerator | None = None,
    ):
        self.settings = settings
        self.registry = registry
        self.cfg = cfg
        self.summary_generator = summary_generator

    def _llm_summary(self, evidence: Mapping[str, Any], draft: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.summary_generator is not None:
            result = self.summary_generator(evidence, draft)
            return self._validate_summary(result, evidence)
        if self.cfg is None:
            return {
                "baseline_summary": str(draft.get("baseline_summary") or "Validated family/hardware baseline."),
                "lesson_summaries": [],
            }
        from llm import FunctionSpec, query

        spec = FunctionSpec(
            name="summarize_lesson_profile_evidence",
            description="Summarize supplied evidence without inventing facts.",
            json_schema={
                "type": "object",
                "properties": {
                    "baseline_summary": {"type": "string"},
                    "lesson_summaries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lesson_type": {"type": "string"},
                                "lesson": {"type": "string"},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["lesson_type", "lesson", "evidence_refs"],
                        },
                    },
                },
                "required": ["baseline_summary", "lesson_summaries"],
            },
        )
        feedback = self.cfg.agent.feedback
        model = self.settings.builder.model or feedback.model
        result = query(
            system_message={
                "Role": "Evidence-bound lesson profile summarizer",
                "Rules": [
                    "Use only the supplied deterministic facts.",
                    "Do not add measurements, paths, credentials, dataset rows, or code.",
                    "Every lesson must cite one or more supplied evidence_refs.",
                    "For multi_change deltas, describe association and never causality.",
                ],
                "Deterministic facts": dict(draft),
                "Evidence references": list(evidence.get("evidence_refs") or []),
            },
            user_message=None,
            func_spec=spec,
            model=model,
            temperature=0.0,
            max_tokens=1200,
            cfg=self.cfg,
        )
        if not isinstance(result, Mapping):
            raise ValueError("Lesson builder returned a non-object response")
        return self._validate_summary(result, evidence)

    def _validate_summary(self, result: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed_refs = set(str(item) for item in evidence.get("evidence_refs") or [])
        known_numbers = _known_numeric_tokens(evidence)
        change_scope = str((evidence.get("delta") or {}).get("change_scope") or "")
        baseline_summary = str(result.get("baseline_summary") or "").strip()
        if len(baseline_summary) > self.settings.builder.max_summary_chars:
            raise ValueError("Builder baseline summary exceeds configured limit")
        for number in _NUMBER.findall(baseline_summary):
            if number not in known_numbers:
                raise ValueError(f"Unsupported numeric claim in builder summary: {number}")
        lessons = result.get("lesson_summaries") or []
        if not isinstance(lessons, list):
            raise ValueError("lesson_summaries must be an array")
        normalized = []
        for item in lessons[:8]:
            if not isinstance(item, Mapping):
                raise ValueError("Every lesson summary must be an object")
            text = str(item.get("lesson") or "").strip()
            refs = [str(ref) for ref in item.get("evidence_refs") or []]
            if not text or not refs or any(ref not in allowed_refs for ref in refs):
                raise ValueError("Builder lesson lacks resolvable evidence references")
            if change_scope == "multi_change" and _CAUSAL_TERMS.search(text):
                raise ValueError("Multi-change observations cannot produce causal claims")
            for number in _NUMBER.findall(text):
                if number not in known_numbers:
                    raise ValueError(f"Unsupported numeric claim in lesson: {number}")
            normalized.append({
                "lesson_type": str(item.get("lesson_type") or "modification"),
                "lesson": text[: self.settings.builder.max_summary_chars],
                "evidence_refs": refs,
            })
        return {"baseline_summary": baseline_summary, "lesson_summaries": normalized}

    def _baseline(
        self,
        evidence: Mapping[str, Any],
        prior: Mapping[str, Any] | None,
        summary: str,
    ) -> dict[str, Any]:
        introspection = dict((evidence.get("code") or {}).get("introspection") or {})
        measurements = dict(evidence.get("scheduler_measurements") or {})
        validation = dict(evidence.get("validation") or {})
        existing = dict((prior or {}).get("baseline") or {})
        baseline = {
            **existing,
            "source_model_key": introspection.get("model_key"),
            "source_code_variant_key": (evidence.get("code") or {}).get("normalized_signature"),
            "structural_fingerprint": ((evidence.get("code") or {}).get("structural") or {}).get("hash"),
            "model_summary": summary,
            "safe_training_defaults": {
                "precision": introspection.get("precision_mode"),
                "physical_batch_size": measurements.get("resolved_batch_size") or introspection.get("batch_size"),
                "gradient_accumulation_steps": introspection.get("gradient_accumulation_steps"),
                "epochs": introspection.get("epochs"),
                "num_workers": introspection.get("num_workers"),
            },
            "resource_envelope": {
                "peak_vram_mb": measurements.get("peak_vram_mb"),
                "observed_runtime_seconds": measurements.get("runtime_seconds") or evidence.get("exec_time"),
                "throughput": measurements.get("throughput"),
            },
            "outcome_summary": {
                "validation_metric": validation.get("metric"),
                "metric_maximize": validation.get("metric_maximize"),
                "source_outcome": evidence.get("outcome"),
            },
            "warnings": list(existing.get("warnings") or []),
        }
        return baseline

    @staticmethod
    def _lesson_shape(evidence: Mapping[str, Any]) -> tuple[str, list[str]]:
        stage = str(evidence.get("stage") or "draft")
        strategy = str(evidence.get("generation_strategy") or "")
        if stage == "debug":
            return "verified_fix", ["debug", "review"]
        if stage == "improve":
            return "modification", ["improve", "review"]
        if stage == "evolution":
            return "branch_trajectory", ["evolution"]
        if stage == "fusion":
            return "transfer", ["fusion", "review"]
        if stage == "fusion_draft" or strategy == "aggregation":
            return "cross_branch_consensus", ["aggregation", "review"]
        return "family_baseline", ["draft"]

    def _new_lessons(
        self,
        evidence: Mapping[str, Any],
        summaries: list[Mapping[str, Any]],
        confidence: float,
        *,
        failure: bool,
    ) -> list[LessonRecord]:
        refs = list(evidence.get("evidence_refs") or [])
        delta = dict(evidence.get("delta") or {})
        signature = hashlib.sha256(
            json.dumps({
                "diff": delta.get("unified_diff") or evidence.get("outcome"),
                "scope": delta.get("change_scope"),
                "action": delta.get("change_action"),
                "layer_type": delta.get("layer_type"),
                "location": delta.get("location_signature"),
            }, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if failure:
            symptom = str((evidence.get("artifacts") or {}).get("terminal_excerpt") or evidence.get("outcome") or "failure")
            return [LessonRecord(
                lesson_id="",
                lesson_type="failure_warning",
                agent_audiences=["debug", "review"],
                content={"lesson": symptom[:700], "outcome": evidence.get("outcome")},
                confidence=min(confidence, 0.65),
                evidence_refs=refs,
                change_signature=signature,
                change_scope=str(delta.get("change_scope") or "training_only"),
                change_action=str(delta.get("change_action") or "other"),
                layer_type=str(delta.get("layer_type") or "other"),
                warnings=["This is a classified failure, not a known-good default."],
            )]
        lesson_type, audiences = self._lesson_shape(evidence)
        matching = next((item for item in summaries if str(item.get("lesson_type")) == lesson_type), None)
        text = str((matching or {}).get("lesson") or "Validated change recorded for this family/hardware profile.")
        content: dict[str, Any] = {
            "lesson": text,
            "observed_outcome": "succeeded",
            "metric": _metric_value(evidence),
            "change_action": delta.get("change_action"),
            "layer_type": delta.get("layer_type"),
            "location_signature": delta.get("location_signature"),
            "training_changes": delta.get("training_changes") or {},
        }
        diff_lines = str(delta.get("unified_diff") or "").splitlines()
        if delta.get("change_scope") != "multi_change" and 0 < len(diff_lines) <= 36:
            content["implementation_example"] = {
                "kind": "minimal_patch",
                "language": "python",
                "code": "\n".join(diff_lines),
                "source": "extracted_from_successful_node",
            }
        warnings = []
        if delta.get("change_scope") == "multi_change":
            warnings.append("Multiple material changes were observed; no causal layer claim is supported.")
        result = [LessonRecord(
            lesson_id="",
            lesson_type=lesson_type,
            agent_audiences=audiences,
            content=content,
            confidence=confidence,
            evidence_refs=list((matching or {}).get("evidence_refs") or refs),
            change_signature=signature,
            change_scope=str(delta.get("change_scope") or "training_only"),
            change_action=str(delta.get("change_action") or "other"),
            layer_type=str(delta.get("layer_type") or "other"),
            warnings=warnings,
        )]
        review_issues = list((evidence.get("artifacts") or {}).get("review_issues") or [])
        checks = []
        for issue in review_issues[:8]:
            if isinstance(issue, Mapping):
                check = issue.get("repair_instruction") or issue.get("evidence")
                if check:
                    checks.append(str(check)[:400])
        if not checks:
            checks = [
                "Preserve the validated data, optimizer, evaluation, and submission interfaces when reusing this change."
            ]
        result.append(LessonRecord(
            lesson_id="",
            lesson_type="implementation_contract",
            agent_audiences=["review"],
            content={"component": lesson_type, "checks": checks},
            confidence=confidence,
            evidence_refs=refs,
            change_signature=signature,
            change_scope=str(delta.get("change_scope") or "training_only"),
            change_action=str(delta.get("change_action") or "other"),
            layer_type=str(delta.get("layer_type") or "other"),
            warnings=warnings,
        ))
        return result

    def build(self, observation: Mapping[str, Any]) -> dict[str, Any] | None:
        evidence = dict(observation["evidence"])
        identity = dict(evidence["identity"])
        profile_key = str(identity["profile_key"])
        outcome = str(observation.get("outcome") or evidence.get("outcome") or "")
        prior = self.registry.active_revision(profile_key)
        if outcome != "valid" and prior is None:
            return None

        delta_scope = str((evidence.get("delta") or {}).get("change_scope") or "")
        structural_hash = str(((evidence.get("code") or {}).get("structural") or {}).get("hash") or "")
        if structural_hash and delta_scope != "multi_change":
            current_success = outcome == "valid"
            for other in self.registry.observations_for_profile(profile_key):
                if other["observation_id"] == observation["observation_id"]:
                    continue
                other_evidence = other["evidence"]
                other_hash = str(((other_evidence.get("code") or {}).get("structural") or {}).get("hash") or "")
                other_scope = str((other_evidence.get("delta") or {}).get("change_scope") or "")
                if other_hash == structural_hash and other_scope != "multi_change" and (other.get("outcome") == "valid") != current_success:
                    self.registry.add_conflict(
                        profile_key=profile_key,
                        claim_key=f"controlled-outcome:{structural_hash}",
                        left_observation_id=str(observation["observation_id"]),
                        right_observation_id=str(other["observation_id"]),
                        details={"reason": "controlled observations have contradictory validation outcomes"},
                    )

        successes = self.registry.distinct_successful_runs(profile_key)
        failures = self.registry.distinct_failed_runs(profile_key)
        conflict_count = self.registry.open_conflict_count(profile_key)
        maturity = "conflicted" if conflict_count else (
            "stable" if successes >= self.settings.stability_threshold else "provisional"
        )
        confidence = _confidence(successes, conflict_count)
        deterministic = {
            "baseline_summary": "A final-validated run supports this exact family, hardware, runtime, backend, and workload scope.",
            "identity": identity,
            "outcome": outcome,
            "validation": evidence.get("validation"),
            "configuration": (evidence.get("code") or {}).get("introspection"),
            "measurements": evidence.get("scheduler_measurements"),
            "delta": evidence.get("delta"),
        }
        summaries = self._llm_summary(evidence, deterministic)
        if outcome == "valid":
            baseline = self._baseline(evidence, prior, str(summaries.get("baseline_summary") or ""))
        else:
            baseline = dict(prior["baseline"])
            warning = f"Observed classified failure: {outcome} ({observation['observation_id']})"
            baseline["warnings"] = list(dict.fromkeys([*(baseline.get("warnings") or []), warning]))[-20:]

        carried: list[dict[str, Any]] = []
        if prior is not None:
            for lesson in self.registry.lessons_for_revision(profile_key, int(prior["revision_number"]))[-48:]:
                carried.append({
                    "lesson_type": lesson["lesson_type"],
                    "agent_audiences": lesson["agent_audiences"],
                    "content": lesson["content"],
                    "change_signature": lesson["change_signature"],
                    "change_scope": lesson["change_scope"],
                    "change_action": lesson["change_action"],
                    "layer_type": lesson["layer_type"],
                    "confidence": lesson["confidence"],
                    "evidence_refs": lesson["evidence_refs"],
                    "warnings": lesson["warnings"],
                })
        new_lessons = self._new_lessons(
            evidence,
            list(summaries.get("lesson_summaries") or []),
            confidence,
            failure=outcome != "valid",
        )
        evidence_refs = list(dict.fromkeys([
            *(prior or {}).get("trust", {}).get("evidence_refs", []),
            *list(evidence.get("evidence_refs") or []),
        ]))[-100:]
        trust = {
            "distinct_successful_runs": successes,
            "distinct_failed_runs": failures,
            "confidence": confidence,
            "conflicts": self.registry.list_conflicts(profile_key),
            "evidence_refs": evidence_refs,
        }
        return {
            "identity": identity,
            "baseline": baseline,
            "trust": trust,
            "maturity": maturity,
            "lessons": [*carried, *new_lessons],
        }
