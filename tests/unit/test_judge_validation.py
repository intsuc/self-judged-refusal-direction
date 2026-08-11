from dataclasses import replace

import pytest

from self_judged_refusal_direction.errors import ArtifactError
from self_judged_refusal_direction.judge_validation import (
    judge_validation_passed,
    validate_judge_validation_results,
)
from self_judged_refusal_direction.schema import (
    JudgeLabel,
    JudgeValidationCase,
    JudgeValidationResult,
)


def test_judge_validation_requires_complete_exact_results() -> None:
    cases = tuple(
        JudgeValidationCase(
            case_id=label.value.lower(),
            original_prompt="prompt",
            thinking_text="thinking",
            final_answer="answer",
            generation_truncated=False,
            expected_label=label,
        )
        for label in JudgeLabel
    )
    results = tuple(
        JudgeValidationResult(
            case_id=case.case_id,
            expected_label=case.expected_label,
            status="OK",
            actual_label=case.expected_label,
        )
        for case in cases
    )

    assert judge_validation_passed(cases, results)
    assert not judge_validation_passed(
        cases,
        (replace(results[0], actual_label=JudgeLabel.UNCERTAIN), *results[1:]),
    )
    with pytest.raises(ArtifactError, match="missing judge validation results"):
        validate_judge_validation_results(cases, results[:-1])
    with pytest.raises(ArtifactError, match="changed expected label"):
        validate_judge_validation_results(
            cases,
            (replace(results[0], expected_label=JudgeLabel.UNCERTAIN), *results[1:]),
        )
