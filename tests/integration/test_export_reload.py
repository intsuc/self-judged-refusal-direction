from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

from self_judged_refusal_direction.ce_loss import compute_ce_loss, raw_text_ce_inputs
from self_judged_refusal_direction.config import ExportConfig, ModelConfig, ProjectConfig, RunConfig
from self_judged_refusal_direction.errors import ArtifactError
from self_judged_refusal_direction.exporting import (
    complete_persisted_deferred_reload,
    export_edited_model,
    export_implementation_hash,
    write_export_manifest,
)
from self_judged_refusal_direction.hashing import object_sha256
from self_judged_refusal_direction.models.gemma4 import Gemma4Adapter
from self_judged_refusal_direction.prompting import judge_profile_hash


class LocalBackend:
    def to_str(self) -> str:
        return '{"model":"local"}'


class LocalTokenizer:
    backend_tokenizer = LocalBackend()

    def __call__(self, _text: str, **_kwargs: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[2, 4, 6, 8, 3]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }


class LocalProcessor:
    tokenizer = LocalTokenizer()

    def to_dict(self) -> dict[str, str]:
        return {"processor_class": "LocalProcessor"}

    def save_pretrained(self, output_dir: str | Path) -> None:
        target = Path(output_dir) / "processor_config.json"
        target.write_text('{"processor_class":"LocalProcessor"}\n', encoding="utf-8")


def tiny_gemma4(final_logit_softcapping: float | None = None) -> Gemma4ForConditionalGeneration:
    text = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=64,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=32,
        tie_word_embeddings=True,
        attention_k_eq_v=False,
        final_logit_softcapping=final_logit_softcapping,
    )
    config = Gemma4Config(
        text_config=text,
        vision_config=None,
        audio_config=None,
        tie_word_embeddings=True,
        image_token_id=30,
        video_token_id=29,
        audio_token_id=28,
    )
    return Gemma4ForConditionalGeneration(config).eval()


@pytest.mark.parametrize("softcap", [None, 4.0])
def test_reference_ce_matches_model_loss(softcap: float | None) -> None:
    torch.manual_seed(7)
    model = tiny_gemma4(softcap)
    processor = LocalProcessor()
    inputs = processor.tokenizer("quality", return_tensors="pt", add_special_tokens=True)
    with torch.inference_mode():
        expected = model(**inputs, labels=inputs["input_ids"], use_cache=False, return_dict=True).loss
    runtime = SimpleNamespace(model=model, adapter=Gemma4Adapter(), processor=processor)
    actual = compute_ce_loss(cast(Any, runtime), raw_text_ce_inputs(cast(Any, runtime), ["quality"]))
    assert actual.mean_loss == pytest.approx(float(expected.item()), rel=2e-6, abs=2e-6)


def test_export_and_fresh_offline_auto_reload(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = tiny_gemma4()
    adapter = Gemma4Adapter()
    direction = torch.tensor([0.2, -0.3, 0.5, 0.1, -0.4, 0.6, -0.2, 0.7])
    plan = adapter.build_weight_edit_plan(model, direction)
    config = ProjectConfig(
        run=RunConfig(output_dir=str(tmp_path / "run")),
        model=ModelConfig(
            id="local/tiny-gemma4",
            revision="a" * 40,
            dtype="float32",
            device_map="cpu",
            attention_implementation="sdpa",
        ),
        export=ExportConfig(
            max_shard_size="10MB",
            edit_chunk_rows=3,
        ),
    )
    probe = {
        "input_ids": torch.tensor([[2, 4, 6, 8], [2, 3, 5, 7]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    output_dir = tmp_path / "exported_model"
    deferred_reload_dir = tmp_path / "deferred_reload"
    validation_hash = "judge-validation-a"
    expected_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    result = export_edited_model(
        model,
        LocalProcessor(),
        adapter,
        plan,
        config,
        probe,
        judge_validation_hash=validation_hash,
        output_dir=output_dir,
        full_validation_metrics={"removal_success_rate": 0.75},
        direction_layer=0,
        probe_atol=3e-5,
        probe_rtol=3e-5,
        orthogonality_atol=3e-5,
        reload_timeout_seconds=60,
        deferred_reload_directory=deferred_reload_dir,
        verify_reload_target_trajectory=False,
    )
    assert result.reload is None
    assert result.deferred_reload is not None
    assert model.model.language_model.embed_tokens.weight is model.lm_head.weight
    assert deferred_reload_dir.stat().st_mode & 0o077 == 0
    assert all(path.stat().st_mode & 0o077 == 0 for path in deferred_reload_dir.iterdir())
    del model
    report = complete_persisted_deferred_reload(deferred_reload_dir)
    completed_manifest = {**result.manifest, "fresh_reload": report.as_dict()}
    write_export_manifest(output_dir, completed_manifest)

    manifest = json.loads((output_dir / "edit_manifest.json").read_text(encoding="utf-8"))
    manifest_body = dict(manifest)
    manifest_digest = manifest_body.pop("manifest_sha256")

    assert report.model_module.startswith("transformers.")
    assert report.parameter_count == expected_parameter_count
    assert report.tied_weights_preserved
    assert tuple(output_dir.glob("*.safetensors"))
    assert not tuple(output_dir.glob("*.bin"))
    assert (output_dir / "processor_config.json").is_file()
    assert manifest["base_revision"] == "a" * 40
    assert manifest["export_implementation_hash"] == export_implementation_hash()
    assert manifest["judge_profile_hash"] == judge_profile_hash()
    assert manifest["judge_validation_hash"] == validation_hash
    assert manifest_digest == object_sha256(manifest_body)
    assert manifest["direction_source_layer"] == 0
    assert manifest["full_validation_metrics"] == {"removal_success_rate": 0.75}
    assert manifest["fresh_reload"]["probe_logits_match"] is True
    assert manifest["temporary_permanent_equivalence"]["passed"] is True
    assert not tuple(output_dir.glob("*.private.jsonl"))

    tensor_path = next(deferred_reload_dir.glob("*.private.safetensors"))
    corrupted = bytearray(tensor_path.read_bytes())
    corrupted[-1] ^= 1
    tensor_path.write_bytes(corrupted)
    with pytest.raises(ArtifactError, match="content hash"):
        complete_persisted_deferred_reload(deferred_reload_dir)
