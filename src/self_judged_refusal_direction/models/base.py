from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from self_judged_refusal_direction.config import ModelConfig, ProjectConfig, TargetGenerationConfig
from self_judged_refusal_direction.errors import CompatibilityError, InvariantError
from self_judged_refusal_direction.hashing import object_sha256
from self_judged_refusal_direction.schema import CompatibilityReport

if TYPE_CHECKING:
    from self_judged_refusal_direction.editing import WeightEditPlan


@dataclass(frozen=True)
class ModuleTarget:
    name: str
    module: nn.Module


@dataclass(frozen=True)
class ResidualWriterTarget(ModuleTarget):
    post_norm_name: str
    post_norm: nn.Module


@dataclass(frozen=True)
class ResponseTokenGrammar:
    thinking_open: tuple[int, ...]
    thinking_close: tuple[int, ...]
    content_closes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ParsedTargetOutput:
    raw_generated_token_ids: tuple[int, ...]
    raw_decoded_output: str
    thinking_text: str
    final_answer: str
    thinking_token_start: int
    thinking_token_end: int
    final_token_start: int
    final_token_end: int
    terminal_found: bool


class ArchitectureAdapter(ABC):
    name: str

    @abstractmethod
    def load_model(self, config: ModelConfig) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def load_processor(self, config: ModelConfig) -> Any:
        raise NotImplementedError

    @abstractmethod
    def text_backbone(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def transformer_blocks(self, model: nn.Module) -> Sequence[nn.Module]:
        raise NotImplementedError

    @abstractmethod
    def hidden_size(self, model: nn.Module) -> int:
        raise NotImplementedError

    @abstractmethod
    def text_embedding(self, model: nn.Module) -> ModuleTarget:
        raise NotImplementedError

    @abstractmethod
    def lm_head(self, model: nn.Module) -> ModuleTarget:
        raise NotImplementedError

    @abstractmethod
    def residual_writers(self, model: nn.Module) -> tuple[ResidualWriterTarget, ...]:
        raise NotImplementedError

    @abstractmethod
    def multimodal_projections(self, model: nn.Module) -> tuple[ModuleTarget, ...]:
        raise NotImplementedError

    @abstractmethod
    def render_target_chat(
        self,
        processor: Any,
        messages: Sequence[Mapping[str, Any]],
        config: TargetGenerationConfig | None = None,
        *,
        thinking_enabled: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def render_judge_chat(
        self,
        processor: Any,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def response_token_grammar(self, processor: Any) -> ResponseTokenGrammar:
        raise NotImplementedError

    @abstractmethod
    def parse_target_trajectory(
        self,
        processor: Any,
        generated_ids: Sequence[int] | torch.Tensor,
        *,
        prefix_ids: Sequence[int] | torch.Tensor = (),
        thinking_enabled: bool = True,
    ) -> ParsedTargetOutput:
        raise NotImplementedError

    def activation_read_points(self, model: nn.Module) -> Sequence[nn.Module]:
        return self.transformer_blocks(model)

    def input_device(self, model: nn.Module) -> torch.device:
        embedding = self.text_embedding(model).module
        weight = getattr(embedding, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
            return weight.device
        for parameter in model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        raise CompatibilityError("model has no materialized input device")

    def context_window(self, model: nn.Module) -> int:
        model_config = getattr(model, "config", None)
        text_config = getattr(model_config, "text_config", None)
        for source in (text_config, model_config):
            value = getattr(source, "max_position_embeddings", None)
            if isinstance(value, int) and value > 0:
                return value
        raise CompatibilityError("model config has no valid context window")

    @abstractmethod
    def dual_direction(self, post_norm: nn.Module, direction: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def build_weight_edit_plan(self, model: nn.Module, direction: torch.Tensor) -> WeightEditPlan:
        raise NotImplementedError

    @abstractmethod
    def compatibility_report(self, model: nn.Module, processor: Any | None = None) -> CompatibilityReport:
        raise NotImplementedError

    def chat_template_hash(self, processor: Any) -> str:
        tokenizer = getattr(processor, "tokenizer", None)
        return object_sha256(
            {
                "chat_template": getattr(processor, "chat_template", None),
                "tokenizer_chat_template": getattr(tokenizer, "chat_template", None),
                "response_template": getattr(tokenizer, "response_template", None),
            }
        )

    def processor_fingerprints(self, processor: Any) -> dict[str, str]:
        if callable(getattr(processor, "apply_chat_template", None)):
            messages = [{"role": "user", "content": "fingerprint"}]
            self.render_target_chat(processor, messages)
            self.render_judge_chat(processor, messages)
        tokenizer = getattr(processor, "tokenizer", None)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        serialize_backend = getattr(backend, "to_str", None)
        if not callable(serialize_backend):
            raise CompatibilityError("processor tokenizer has no serializable backend")
        processor_to_dict = getattr(processor, "to_dict", None)
        if not callable(processor_to_dict):
            raise CompatibilityError("processor has no serializable configuration")
        processor_config = processor_to_dict()
        if not isinstance(processor_config, Mapping):
            raise CompatibilityError("processor configuration is not a mapping")
        return {
            "tokenizer_sha256": object_sha256(
                {
                    "class": type(tokenizer).__name__,
                    "backend": serialize_backend(),
                    "chat_template": getattr(tokenizer, "chat_template", None),
                    "response_template": getattr(tokenizer, "response_template", None),
                }
            ),
            "processor_sha256": object_sha256(
                {
                    "class": type(processor).__name__,
                    "config": processor_config,
                    "chat_template": getattr(processor, "chat_template", None),
                }
            ),
        }

    def save_model_and_processor(
        self,
        model: nn.Module,
        processor: Any,
        output_dir: str | Path,
        config: ProjectConfig,
    ) -> None:
        target = Path(output_dir)
        save_model = getattr(model, "save_pretrained", None)
        if not callable(save_model):
            raise InvariantError("model does not support save_pretrained")
        save_model(
            target,
            safe_serialization=True,
            max_shard_size=config.export.max_shard_size,
        )
        save_processor = getattr(processor, "save_pretrained", None)
        if not callable(save_processor):
            raise InvariantError("processor does not support save_pretrained")
        save_processor(target)
