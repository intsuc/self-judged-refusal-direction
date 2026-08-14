from dataclasses import replace

import torch

from self_judged_refusal_direction.artifacts import ArtifactProfile
from self_judged_refusal_direction.config import DataConfig, ModelConfig, ProjectConfig, RunConfig
from self_judged_refusal_direction.directions import CandidateBundle
from self_judged_refusal_direction.pipeline import (
    _candidate_evaluation_hash,
    _candidate_kl_screen_identity,
    _candidate_phase_identity,
    _candidate_reference_ce_identity,
)
from self_judged_refusal_direction.schema import (
    DirectionCandidate,
    LabeledTrajectory,
    PromptRecord,
    TargetTrajectory,
)


class Adapter:
    pass


def test_reference_files_only_invalidate_reference_ce_identity(tmp_path, monkeypatch) -> None:
    first_reference = tmp_path / "first.txt"
    second_reference = tmp_path / "second.txt"
    first_reference.write_text("first reference\n", encoding="utf-8")
    second_reference.write_text("second reference\n", encoding="utf-8")
    base = ProjectConfig(
        run=RunConfig(output_dir=str(tmp_path / "run")),
        model=ModelConfig(id="model", revision="a" * 40),
        data=DataConfig(reference_files=(str(first_reference),)),
    )
    changed = replace(base, data=replace(base.data, reference_files=(str(second_reference),)))
    monkeypatch.setattr(
        "self_judged_refusal_direction.pipeline.adapter_for_config",
        lambda _config: Adapter(),
    )
    candidate = DirectionCandidate(
        candidate_id="candidate",
        layer=1,
        norm=1.0,
        refusal_count=2,
        non_refusal_count=2,
        standardized_separation=1.0,
        refusal_projected_mean=1.0,
        non_refusal_projected_mean=0.0,
        refusal_projected_variance_diagonal=1.0,
        non_refusal_projected_variance_diagonal=1.0,
        finite=True,
    )
    bundle = CandidateBundle(directions=torch.tensor([[1.0, 0.0]]), candidates=(candidate,))
    prompt = PromptRecord(prompt_id="prompt", original_prompt="prompt", group_id="group", split="validation")
    trajectory = TargetTrajectory(
        prompt_id=prompt.prompt_id,
        original_prompt=prompt.original_prompt,
        raw_generated_token_ids=(1, 2),
        raw_decoded_output="answer",
        thinking_text="",
        final_answer="answer",
        thinking_token_start=0,
        thinking_token_end=0,
        final_token_start=0,
        final_token_end=2,
        generation_truncated=False,
        parser_status="OK",
        model_revision=base.model.revision,
        generation_config_hash="b" * 64,
        trajectory_hash="c" * 64,
        split="validation",
    )
    labeled = LabeledTrajectory(
        prompt_id=prompt.prompt_id,
        label="NON_REFUSAL",
        trajectory_hash=trajectory.trajectory_hash,
    )
    profile = ArtifactProfile(
        model_id="model",
        model_revision=base.model.revision,
        artifact_stage="candidate_evaluation",
        stage_config_hash=base.stage_config_hash("candidate_evaluation"),
        candidate_evaluation_hash="d" * 64,
    )

    def identities(config: ProjectConfig) -> tuple[str, str, str, str]:
        return (
            _candidate_evaluation_hash(config),
            _candidate_kl_screen_identity(config, bundle, (candidate,), (prompt,), profile),
            _candidate_phase_identity(
                config,
                bundle,
                (candidate,),
                (labeled,),
                {prompt.prompt_id: prompt},
                {prompt.prompt_id: trajectory},
                (prompt,),
                profile,
                "pilot_evaluation",
                {candidate.candidate_id: 0.1},
            ),
            _candidate_reference_ce_identity(
                config,
                bundle,
                (candidate,),
                profile,
                "pilot_evaluation",
            ),
        )

    base_identities = identities(base)
    changed_identities = identities(changed)

    assert base_identities[:3] == changed_identities[:3]
    assert base_identities[3] != changed_identities[3]
