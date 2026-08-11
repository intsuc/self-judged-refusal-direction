from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.resources import files
from typing import Any

from tqdm import tqdm

from self_judged_refusal_direction.errors import ArtifactError, ConfigurationError, InvariantError
from self_judged_refusal_direction.hashing import object_sha256
from self_judged_refusal_direction.judging import TrajectoryJudge
from self_judged_refusal_direction.runtime import BaseModelRuntime
from self_judged_refusal_direction.schema import (
    JudgeInput,
    JudgeLabel,
    JudgeValidationCase,
    JudgeValidationResult,
)

_FIXTURE_NAME = "judge_semantics.jsonl"
_FIXTURE_FIELDS = {
    "case_id",
    "original_prompt",
    "thinking_text",
    "final_answer",
    "generation_truncated",
    "expected_label",
}
_FIXTURE_STRING_FIELDS = {
    "case_id",
    "original_prompt",
    "thinking_text",
    "final_answer",
    "expected_label",
}


class _DuplicateJsonKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def load_judge_validation_cases() -> tuple[JudgeValidationCase, ...]:
    resource = files("self_judged_refusal_direction").joinpath(_FIXTURE_NAME)
    try:
        fixture = resource.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigurationError("judge validation fixture is unavailable or is not UTF-8") from error
    lines = fixture.splitlines()
    if not lines:
        raise ConfigurationError("judge validation fixture is empty")
    cases: list[JudgeValidationCase] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ConfigurationError(f"judge validation fixture line {line_number} is empty")
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, _DuplicateJsonKey) as error:
            raise ConfigurationError(f"judge validation fixture line {line_number} is invalid JSON") from error
        if not isinstance(value, dict) or set(value) != _FIXTURE_FIELDS:
            raise ConfigurationError(f"judge validation fixture line {line_number} has invalid fields")
        if any(type(value[field]) is not str for field in _FIXTURE_STRING_FIELDS):
            raise ConfigurationError(f"judge validation fixture line {line_number} has an invalid string field")
        if type(value["generation_truncated"]) is not bool:
            raise ConfigurationError(f"judge validation fixture line {line_number} has an invalid truncation flag")
        case_id = value["case_id"]
        if not case_id:
            raise ConfigurationError(f"judge validation fixture line {line_number} has an empty case_id")
        if case_id in case_ids:
            raise ConfigurationError(f"judge validation fixture has duplicate case_id: {case_id}")
        try:
            expected_label = JudgeLabel(value["expected_label"])
        except ValueError as error:
            raise ConfigurationError(f"judge validation fixture line {line_number} has an invalid label") from error
        case_ids.add(case_id)
        cases.append(
            JudgeValidationCase(
                case_id=case_id,
                original_prompt=value["original_prompt"],
                thinking_text=value["thinking_text"],
                final_answer=value["final_answer"],
                generation_truncated=value["generation_truncated"],
                expected_label=expected_label,
            )
        )
    observed_labels = {case.expected_label for case in cases}
    missing_labels = set(JudgeLabel) - observed_labels
    if missing_labels:
        names = ", ".join(sorted(label.value for label in missing_labels))
        raise ConfigurationError(f"judge validation fixture is missing labels: {names}")
    return tuple(cases)


def judge_validation_fixture_hash(cases: Sequence[JudgeValidationCase]) -> str:
    return object_sha256(tuple(case.as_dict() for case in cases))


def run_judge_validation(
    runtime: BaseModelRuntime,
    cases: Sequence[JudgeValidationCase],
) -> tuple[JudgeValidationResult, ...]:
    judge = TrajectoryJudge(runtime.adapter, runtime.model, runtime.processor)
    results: list[JudgeValidationResult] = []
    for case in tqdm(
        cases,
        desc="Validating judge",
        unit="case",
        dynamic_ncols=True,
        disable=None,
    ):
        input_hash = object_sha256(
            {
                "case_id": case.case_id,
                "original_prompt": case.original_prompt,
                "thinking_text": case.thinking_text,
                "final_answer": case.final_answer,
                "generation_truncated": case.generation_truncated,
            }
        )
        result = judge.classify_input(
            JudgeInput(
                original_prompt=case.original_prompt,
                thinking_text=case.thinking_text,
                final_answer=case.final_answer,
                generation_truncated=case.generation_truncated,
                input_hash=input_hash,
            )
        )
        if result.status == "OK":
            if result.label is None:
                raise InvariantError("successful judge validation result has no label")
            results.append(
                JudgeValidationResult(
                    case_id=case.case_id,
                    expected_label=case.expected_label,
                    status="OK",
                    actual_label=JudgeLabel(result.label),
                )
            )
        else:
            if not result.error_code:
                raise InvariantError("failed judge validation result has no error code")
            results.append(
                JudgeValidationResult(
                    case_id=case.case_id,
                    expected_label=case.expected_label,
                    status="ERROR",
                    error_code=result.error_code,
                )
            )
    return tuple(results)


def validate_judge_validation_results(
    cases: Sequence[JudgeValidationCase],
    results: Sequence[JudgeValidationResult],
) -> None:
    if not cases:
        raise ArtifactError("judge validation cases are empty")
    case_by_id: dict[str, JudgeValidationCase] = {}
    for case in cases:
        if type(case.case_id) is not str or not case.case_id or not isinstance(case.expected_label, JudgeLabel):
            raise ArtifactError("invalid judge validation case")
        if case.case_id in case_by_id:
            raise ArtifactError(f"duplicate judge validation case: {case.case_id}")
        case_by_id[case.case_id] = case
    result_by_id: dict[str, JudgeValidationResult] = {}
    for result in results:
        if result.case_id in result_by_id:
            raise ArtifactError(f"duplicate judge validation result: {result.case_id}")
        if result.case_id not in case_by_id:
            raise ArtifactError(f"unknown judge validation result: {result.case_id}")
        if not isinstance(result.expected_label, JudgeLabel):
            raise ArtifactError(f"invalid expected label for judge validation result: {result.case_id}")
        if result.expected_label != case_by_id[result.case_id].expected_label:
            raise ArtifactError(f"changed expected label for judge validation result: {result.case_id}")
        if result.status == "OK":
            if not isinstance(result.actual_label, JudgeLabel) or result.error_code is not None:
                raise ArtifactError(f"invalid successful judge validation result: {result.case_id}")
        elif result.status == "ERROR":
            if result.actual_label is not None or type(result.error_code) is not str or not result.error_code:
                raise ArtifactError(f"invalid failed judge validation result: {result.case_id}")
        else:
            raise ArtifactError(f"invalid judge validation status: {result.case_id}")
        result_by_id[result.case_id] = result
    missing = set(case_by_id) - set(result_by_id)
    if missing:
        raise ArtifactError(f"missing judge validation results: {', '.join(sorted(missing))}")


def judge_validation_passed(
    cases: Sequence[JudgeValidationCase],
    results: Sequence[JudgeValidationResult],
) -> bool:
    validate_judge_validation_results(cases, results)
    return all(result.status == "OK" and result.actual_label == result.expected_label for result in results)


__all__ = [
    "judge_validation_fixture_hash",
    "judge_validation_passed",
    "load_judge_validation_cases",
    "run_judge_validation",
    "validate_judge_validation_results",
]
