from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from subprocess import SubprocessError
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForMultimodalLM, AutoProcessor

from self_judged_refusal_direction.config import ModelConfig, TargetGenerationConfig
from self_judged_refusal_direction.errors import (
    CompatibilityError,
    InvariantError,
    TargetParseError,
    TargetParseErrorCode,
)
from self_judged_refusal_direction.hashing import object_sha256
from self_judged_refusal_direction.models.base import (
    ArchitectureAdapter,
    ModuleTarget,
    ParsedTargetOutput,
    ResidualWriterTarget,
    ResponseTokenGrammar,
)
from self_judged_refusal_direction.schema import CompatibilityReport


@dataclass(frozen=True)
class Gemma4Topology:
    text_backbone: nn.Module
    blocks: tuple[nn.Module, ...]
    embedding: ModuleTarget
    lm_head: ModuleTarget
    residual_writers: tuple[ResidualWriterTarget, ...]
    multimodal_projections: tuple[ModuleTarget, ...]
    final_norm: ModuleTarget
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    moe_enabled: bool
    ple_enabled: bool
    tied_embeddings: bool


def _weight_shape(module: nn.Module, name: str) -> tuple[int, ...]:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise CompatibilityError(f"{name} does not expose a tensor weight")
    return tuple(weight.shape)


def _token_ids(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        data = value.detach().to(device="cpu").tolist()
    elif hasattr(value, "tolist") and not isinstance(value, list | tuple):
        data = value.tolist()
    else:
        data = value
    if isinstance(data, list | tuple) and len(data) == 1 and isinstance(data[0], list | tuple):
        data = data[0]
    if not isinstance(data, list | tuple) or any(not isinstance(item, int) for item in data):
        raise InvariantError(f"{name} must contain exactly one token sequence")
    return tuple(data)


def _find_sequence(tokens: tuple[int, ...], pattern: tuple[int, ...], start: int) -> int | None:
    if not pattern:
        return None
    limit = len(tokens) - len(pattern) + 1
    for index in range(start, limit):
        if tokens[index : index + len(pattern)] == pattern:
            return index
    return None


def _decode(processor: Any, token_ids: Sequence[int]) -> str:
    try:
        value = processor.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        value = processor.decode(list(token_ids), skip_special_tokens=False)
    if not isinstance(value, str):
        raise InvariantError("processor.decode did not return one string")
    return value


class Gemma4Adapter(ArchitectureAdapter):
    name = "gemma4"

    def load_model(self, config: ModelConfig) -> nn.Module:
        if not isinstance(config.id, str) or not config.id:
            raise InvariantError("model.id is required")
        dtype = getattr(torch, config.dtype, None)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise InvariantError(f"unsupported model dtype: {config.dtype}")
        return AutoModelForMultimodalLM.from_pretrained(
            config.id,
            revision=config.revision,
            trust_remote_code=False,
            dtype=dtype,
            device_map=config.device_map,
            low_cpu_mem_usage=True,
            attn_implementation=config.attention_implementation,
        )

    def load_processor(self, config: ModelConfig) -> Any:
        if not isinstance(config.id, str) or not config.id:
            raise InvariantError("model.id is required")
        return AutoProcessor.from_pretrained(
            config.id,
            revision=config.revision,
            trust_remote_code=False,
        )

    def text_backbone(self, model: nn.Module) -> nn.Module:
        return self._topology(model).text_backbone

    def transformer_blocks(self, model: nn.Module) -> Sequence[nn.Module]:
        return self._topology(model).blocks

    def hidden_size(self, model: nn.Module) -> int:
        return self._topology(model).hidden_size

    def text_embedding(self, model: nn.Module) -> ModuleTarget:
        return self._topology(model).embedding

    def lm_head(self, model: nn.Module) -> ModuleTarget:
        return self._topology(model).lm_head

    def residual_writers(self, model: nn.Module) -> tuple[ResidualWriterTarget, ...]:
        return self._topology(model).residual_writers

    def multimodal_projections(self, model: nn.Module) -> tuple[ModuleTarget, ...]:
        return self._topology(model).multimodal_projections

    def render_target_chat(
        self,
        processor: Any,
        messages: Sequence[Mapping[str, Any]],
        config: TargetGenerationConfig | None = None,
        *,
        thinking_enabled: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        configured = config.thinking_enabled if config is not None else None
        if configured is not None and not isinstance(configured, bool):
            raise InvariantError("target thinking mode must be a boolean")
        if thinking_enabled is None:
            resolved = True if configured is None else configured
        else:
            if not isinstance(thinking_enabled, bool):
                raise InvariantError("target thinking mode must be a boolean")
            if configured is not None and thinking_enabled is not configured:
                raise InvariantError("explicit target thinking mode conflicts with generation config")
            resolved = thinking_enabled
        return self._render_chat(processor, messages, enable_thinking=resolved, **kwargs)

    def render_judge_chat(
        self,
        processor: Any,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Any:
        return self._render_chat(processor, messages, enable_thinking=False, **kwargs)

    def response_token_grammar(self, processor: Any) -> ResponseTokenGrammar:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise CompatibilityError("processor has no tokenizer")
        response_template = getattr(tokenizer, "response_template", None)
        if not isinstance(response_template, Mapping):
            raise CompatibilityError("processor tokenizer has no response_template")
        fields = response_template.get("fields")
        if not isinstance(fields, Mapping):
            raise CompatibilityError("response_template has no fields mapping")
        thinking = fields.get("thinking")
        content = fields.get("content")
        if not isinstance(thinking, Mapping) or not isinstance(content, Mapping):
            raise CompatibilityError("response_template must define thinking and content fields")
        thinking_open = thinking.get("open")
        thinking_close = thinking.get("close")
        content_closes = content.get("close")
        if not isinstance(thinking_open, str) or not isinstance(thinking_close, str):
            raise CompatibilityError("thinking response delimiters must be literal strings")
        if isinstance(content_closes, str):
            content_closes = [content_closes]
        if (
            not isinstance(content_closes, list | tuple)
            or not content_closes
            or any(not isinstance(item, str) for item in content_closes)
        ):
            raise CompatibilityError("content response delimiters must be literal strings")
        tokenizer_encode = getattr(tokenizer, "encode", None)
        if not callable(tokenizer_encode):
            raise CompatibilityError("processor tokenizer has no encode method")

        def encode(value: str) -> tuple[int, ...]:
            encoded = tokenizer_encode(value, add_special_tokens=False)
            result = _token_ids(encoded, "response delimiter")
            if not result:
                raise CompatibilityError("response delimiter tokenized to an empty sequence")
            return result

        return ResponseTokenGrammar(
            thinking_open=encode(thinking_open),
            thinking_close=encode(thinking_close),
            content_closes=tuple(encode(item) for item in content_closes),
        )

    def parse_target_trajectory(
        self,
        processor: Any,
        generated_ids: Sequence[int] | torch.Tensor,
        *,
        prefix_ids: Sequence[int] | torch.Tensor = (),
        thinking_enabled: bool = True,
    ) -> ParsedTargetOutput:
        if not isinstance(thinking_enabled, bool):
            raise TargetParseError(TargetParseErrorCode.INVALID_MODE, "target thinking mode must be a boolean")
        try:
            tokens = _token_ids(generated_ids, "generated_ids")
            prefix = _token_ids(prefix_ids, "prefix_ids")
        except ImportError, MemoryError, OSError, RuntimeError, SubprocessError:
            raise
        except Exception as error:
            raise TargetParseError(
                TargetParseErrorCode.INVALID_INPUT,
                f"{type(error).__name__}: {error}",
            ) from error
        try:
            grammar = self.response_token_grammar(processor)
        except ImportError, MemoryError, OSError, RuntimeError, SubprocessError:
            raise
        except Exception as error:
            raise TargetParseError(
                TargetParseErrorCode.INVALID_GRAMMAR,
                f"{type(error).__name__}: {error}",
            ) from error
        if thinking_enabled:
            if tokens[: len(grammar.thinking_open)] != grammar.thinking_open:
                raise TargetParseError(
                    TargetParseErrorCode.THINKING_OPEN_MISSING,
                    "thinking output does not start with the official response delimiter",
                )
            thinking_start = len(grammar.thinking_open)
            thinking_end = _find_sequence(tokens, grammar.thinking_close, thinking_start)
            if thinking_end is None:
                raise TargetParseError(
                    TargetParseErrorCode.THINKING_CLOSE_MISSING,
                    "thinking output has no official closing delimiter",
                )
            final_start = thinking_end + len(grammar.thinking_close)
        else:
            if _find_sequence(tokens, grammar.thinking_open, 0) is not None:
                raise TargetParseError(
                    TargetParseErrorCode.THINKING_DELIMITER_IN_CONTENT,
                    "content-only output contains a thinking opening delimiter",
                )
            if _find_sequence(tokens, grammar.thinking_close, 0) is not None:
                raise TargetParseError(
                    TargetParseErrorCode.THINKING_DELIMITER_IN_CONTENT,
                    "content-only output contains a thinking closing delimiter",
                )
            thinking_start = 0
            thinking_end = 0
            final_start = 0

        terminal_candidates: list[tuple[int, tuple[int, ...]]] = []
        for pattern in grammar.content_closes:
            index = _find_sequence(tokens, pattern, final_start)
            if index is not None:
                terminal_candidates.append((index, pattern))
        if terminal_candidates:
            terminal_start, terminal_pattern = min(terminal_candidates, key=lambda item: (item[0], -len(item[1])))
            terminal_end = terminal_start + len(terminal_pattern)
            pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
            trailing = tokens[terminal_end:]
            if trailing and (pad_token_id is None or any(token != pad_token_id for token in trailing)):
                raise TargetParseError(
                    TargetParseErrorCode.TRAILING_TOKENS,
                    "generated output continues after its terminal delimiter",
                )
            final_end = terminal_start
            terminal_found = True
        else:
            if not thinking_enabled:
                raise TargetParseError(
                    TargetParseErrorCode.TERMINAL_MISSING,
                    "content-only output has no official terminal delimiter",
                )
            final_end = len(tokens)
            terminal_found = False

        try:
            parsed = processor.parse_response(list(tokens), prefix=list(prefix))
        except ImportError, MemoryError, OSError, RuntimeError, SubprocessError:
            raise
        except Exception as error:
            raise TargetParseError(
                TargetParseErrorCode.OFFICIAL_PARSER_REJECTED,
                f"official response parser rejected generated output: {type(error).__name__}: {error}",
            ) from error
        if isinstance(parsed, list) and len(parsed) == 1:
            parsed = parsed[0]
        if not isinstance(parsed, Mapping):
            raise TargetParseError(
                TargetParseErrorCode.OFFICIAL_RESPONSE_INVALID,
                "official response parser did not return one message",
            )
        thinking_text = parsed.get("thinking", "")
        final_answer = parsed.get("content", "")
        if not isinstance(thinking_text, str) or not isinstance(final_answer, str):
            raise TargetParseError(
                TargetParseErrorCode.RESPONSE_FIELDS_INVALID,
                "official response parser returned non-text response fields",
            )
        if not thinking_enabled and thinking_text:
            raise TargetParseError(
                TargetParseErrorCode.UNEXPECTED_THINKING,
                "official response parser found thinking in content-only output",
            )
        try:
            raw_decoded_output = _decode(processor, tokens)
            decoded_final = _decode(processor, tokens[final_start:final_end])
            if thinking_enabled:
                decoded_thinking = _decode(processor, tokens[thinking_start:thinking_end])
                parser_disagrees = decoded_thinking.strip() != thinking_text or decoded_final.strip() != final_answer
            else:
                parser_disagrees = decoded_final.strip() != final_answer
        except ImportError, MemoryError, OSError, RuntimeError, SubprocessError:
            raise
        except Exception as error:
            raise TargetParseError(
                TargetParseErrorCode.DECODE_FAILED,
                f"{type(error).__name__}: {error}",
            ) from error
        if parser_disagrees:
            raise TargetParseError(
                TargetParseErrorCode.BOUNDARY_MISMATCH,
                "official response parser disagrees with token grammar boundaries",
            )

        return ParsedTargetOutput(
            raw_generated_token_ids=tokens,
            raw_decoded_output=raw_decoded_output,
            thinking_text=thinking_text,
            final_answer=final_answer,
            thinking_token_start=thinking_start,
            thinking_token_end=thinking_end,
            final_token_start=final_start,
            final_token_end=final_end,
            terminal_found=terminal_found,
        )

    def dual_direction(self, post_norm: nn.Module, direction: torch.Tensor) -> torch.Tensor:
        if direction.ndim != 1:
            raise InvariantError("direction must be rank one")
        compute_dtype = torch.float64 if direction.dtype == torch.float64 else torch.float32
        dual = direction.detach().to(dtype=compute_dtype)
        if getattr(post_norm, "with_scale", True):
            weight = getattr(post_norm, "weight", None)
            if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != tuple(direction.shape):
                raise CompatibilityError("post-residual norm scale is incompatible with direction")
            dual = dual * weight.detach().to(device=dual.device, dtype=compute_dtype)
        norm = torch.linalg.vector_norm(dual)
        if not bool(torch.isfinite(norm)) or norm.item() == 0:
            raise InvariantError("dual direction has zero or non-finite norm")
        return dual / norm

    def build_weight_edit_plan(self, model: nn.Module, direction: torch.Tensor) -> Any:
        from self_judged_refusal_direction.editing import WeightEditPlan

        return WeightEditPlan.from_adapter(adapter=self, model=model, direction=direction)

    def compatibility_report(self, model: nn.Module, processor: Any | None = None) -> CompatibilityReport:
        parameter_shapes = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        architecture = tuple(getattr(getattr(model, "config", None), "architectures", ()) or ())
        if not architecture:
            architecture = (type(model).__name__,)
        errors: list[str] = []
        topology_data: dict[str, Any] = {}
        try:
            topology = self._topology(model)
        except CompatibilityError as error:
            errors.append(str(error))
            topology = None

        model_type = getattr(getattr(model, "config", None), "model_type", None)
        if model_type != "gemma4":
            errors.append(f"expected model_type gemma4, found {model_type!r}")
        if processor is not None:
            try:
                grammar = self.response_token_grammar(processor)
                if not callable(getattr(processor, "parse_response", None)):
                    raise CompatibilityError("processor has no parse_response method")
                probe = (*grammar.thinking_open, *grammar.thinking_close, *grammar.content_closes[0])
                self.parse_target_trajectory(processor, probe)
                self.parse_target_trajectory(
                    processor,
                    grammar.content_closes[0],
                    thinking_enabled=False,
                )
            except (CompatibilityError, InvariantError) as error:
                errors.append(str(error))

        if topology is not None:
            if topology.moe_enabled:
                errors.append("Gemma 4 MoE export is not supported")
            if topology.ple_enabled:
                errors.append("Gemma 4 per-layer embedding export is not supported")
            if not topology.tied_embeddings:
                errors.append("Gemma 4 input embedding and LM head are not tied")
            config = getattr(model, "config", None)
            if getattr(config, "quantization_config", None) is not None:
                errors.append("quantized Gemma 4 checkpoints are not supported")
            vision_config = getattr(config, "vision_config", None)
            if vision_config is not None and not topology.multimodal_projections:
                errors.append("vision tower has no compatible text-space projection")
            audio_config = getattr(config, "audio_config", None)
            if audio_config is not None and not any(
                target.name.startswith("model.embed_audio.") for target in topology.multimodal_projections
            ):
                errors.append("audio tower has no compatible text-space projection")
            topology_data = {
                "text_backbone": "model.language_model",
                "blocks": "model.language_model.layers",
                "embedding": topology.embedding.name,
                "lm_head": topology.lm_head.name,
                "final_norm": topology.final_norm.name,
                "residual_writers": [
                    {
                        "name": target.name,
                        "shape": _weight_shape(target.module, target.name),
                        "post_norm": target.post_norm_name,
                    }
                    for target in topology.residual_writers
                ],
                "multimodal_projections": [
                    {"name": target.name, "shape": _weight_shape(target.module, target.name)}
                    for target in topology.multimodal_projections
                ],
                "moe_enabled": topology.moe_enabled,
                "ple_enabled": topology.ple_enabled,
                "tied_embeddings": topology.tied_embeddings,
            }

        return CompatibilityReport(
            adapter=self.name,
            model_class=type(model).__name__,
            architecture=architecture,
            hidden_size=topology.hidden_size if topology is not None else 0,
            num_hidden_layers=topology.num_hidden_layers if topology is not None else 0,
            vocab_size=topology.vocab_size if topology is not None else 0,
            parameter_count=parameter_count,
            parameter_shapes_hash=object_sha256(parameter_shapes),
            compatible=not errors,
            errors=tuple(errors),
            topology=topology_data,
        )

    def _render_chat(
        self,
        processor: Any,
        messages: Sequence[Mapping[str, Any]],
        *,
        enable_thinking: bool,
        **kwargs: Any,
    ) -> Any:
        if "thinking_enabled" in kwargs:
            raise InvariantError("Gemma 4 chat templates use enable_thinking")
        requested = kwargs.pop("enable_thinking", enable_thinking)
        if requested is not enable_thinking:
            raise InvariantError("attempted to override the required thinking mode")
        options = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
        }
        options.update(kwargs)
        options["enable_thinking"] = enable_thinking
        return processor.apply_chat_template(list(messages), **options)

    def _topology(self, model: nn.Module) -> Gemma4Topology:
        container = getattr(model, "model", None)
        if not isinstance(container, nn.Module):
            raise CompatibilityError("Gemma 4 conditional model has no model module")
        text_backbone = getattr(container, "language_model", None)
        if not isinstance(text_backbone, nn.Module):
            raise CompatibilityError("Gemma 4 conditional model has no language_model module")
        raw_blocks = getattr(text_backbone, "layers", None)
        if raw_blocks is None:
            raise CompatibilityError("Gemma 4 language model has no layer sequence")
        try:
            blocks = tuple(raw_blocks)
        except TypeError as error:
            raise CompatibilityError("Gemma 4 language model has no iterable layer sequence") from error
        if not blocks or any(not isinstance(block, nn.Module) for block in blocks):
            raise CompatibilityError("Gemma 4 language model has an invalid layer sequence")

        embedding_module = getattr(text_backbone, "embed_tokens", None)
        head_module = getattr(model, "lm_head", None)
        final_norm_module = getattr(text_backbone, "norm", None)
        if (
            not isinstance(embedding_module, nn.Module)
            or not isinstance(head_module, nn.Module)
            or not isinstance(final_norm_module, nn.Module)
        ):
            raise CompatibilityError("Gemma 4 text embedding, final norm, or LM head is missing")
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        hidden_size = int(getattr(text_config, "hidden_size", 0))
        vocab_size = int(getattr(text_config, "vocab_size", 0))
        num_hidden_layers = int(getattr(text_config, "num_hidden_layers", 0))
        if hidden_size < 1 or vocab_size < 1:
            raise CompatibilityError("Gemma 4 text config has invalid dimensions")
        if num_hidden_layers < 1 or len(blocks) != num_hidden_layers:
            raise CompatibilityError("Gemma 4 layer sequence does not match config")
        if _weight_shape(embedding_module, "model.language_model.embed_tokens") != (vocab_size, hidden_size):
            raise CompatibilityError("Gemma 4 text embedding shape does not match config")
        if _weight_shape(head_module, "lm_head") != (vocab_size, hidden_size):
            raise CompatibilityError("Gemma 4 LM head shape does not match config")
        if _weight_shape(final_norm_module, "model.language_model.norm") != (hidden_size,):
            raise CompatibilityError("Gemma 4 final norm shape does not match config")
        if type(final_norm_module).__name__ != "Gemma4RMSNorm":
            raise CompatibilityError("Gemma 4 uses an unsupported final norm")

        writers: list[ResidualWriterTarget] = []
        for index, block in enumerate(blocks):
            attention = getattr(block, "self_attn", None)
            mlp = getattr(block, "mlp", None)
            attention_writer = getattr(attention, "o_proj", None)
            mlp_writer = getattr(mlp, "down_proj", None)
            attention_norm = getattr(block, "post_attention_layernorm", None)
            mlp_norm = getattr(block, "post_feedforward_layernorm", None)
            if (
                not isinstance(attention_writer, nn.Module)
                or not isinstance(mlp_writer, nn.Module)
                or not isinstance(attention_norm, nn.Module)
                or not isinstance(mlp_norm, nn.Module)
            ):
                raise CompatibilityError(f"Gemma 4 layer {index} has an unsupported residual topology")
            attention_name = f"model.language_model.layers.{index}.self_attn.o_proj"
            mlp_name = f"model.language_model.layers.{index}.mlp.down_proj"
            attention_norm_name = f"model.language_model.layers.{index}.post_attention_layernorm"
            mlp_norm_name = f"model.language_model.layers.{index}.post_feedforward_layernorm"
            attention_shape = _weight_shape(attention_writer, attention_name)
            mlp_shape = _weight_shape(mlp_writer, mlp_name)
            if len(attention_shape) != 2 or attention_shape[0] != hidden_size:
                raise CompatibilityError(f"Gemma 4 layer {index} attention writer has the wrong output size")
            if len(mlp_shape) != 2 or mlp_shape[0] != hidden_size:
                raise CompatibilityError(f"Gemma 4 layer {index} MLP writer has the wrong output size")
            if _weight_shape(attention_norm, attention_norm_name) != (hidden_size,):
                raise CompatibilityError(f"Gemma 4 layer {index} attention norm has the wrong shape")
            if _weight_shape(mlp_norm, mlp_norm_name) != (hidden_size,):
                raise CompatibilityError(f"Gemma 4 layer {index} MLP norm has the wrong shape")
            if type(attention_norm).__name__ != "Gemma4RMSNorm" or type(mlp_norm).__name__ != "Gemma4RMSNorm":
                raise CompatibilityError(f"Gemma 4 layer {index} uses an unsupported post-residual norm")
            writers.extend(
                (
                    ResidualWriterTarget(
                        name=attention_name,
                        module=attention_writer,
                        post_norm_name=attention_norm_name,
                        post_norm=attention_norm,
                    ),
                    ResidualWriterTarget(
                        name=mlp_name,
                        module=mlp_writer,
                        post_norm_name=mlp_norm_name,
                        post_norm=mlp_norm,
                    ),
                )
            )

        multimodal: list[ModuleTarget] = []
        for attribute, name in (
            ("embed_vision", "model.embed_vision.embedding_projection"),
            ("embed_audio", "model.embed_audio.embedding_projection"),
        ):
            embedder = getattr(container, attribute, None)
            if embedder is None:
                continue
            projection = getattr(embedder, "embedding_projection", None)
            if not isinstance(projection, nn.Module):
                raise CompatibilityError(f"{name} is not a text-space projection")
            projection_shape = _weight_shape(projection, name)
            if len(projection_shape) != 2 or projection_shape[0] != hidden_size:
                raise CompatibilityError(f"{name} is not a text-space projection")
            multimodal.append(ModuleTarget(name=name, module=projection))

        moe_enabled = bool(getattr(text_config, "enable_moe_block", False)) or any(
            bool(getattr(block, "enable_moe_block", False)) for block in blocks
        )
        ple_enabled = int(getattr(text_config, "hidden_size_per_layer_input", 0) or 0) > 0 or any(
            int(getattr(block, "hidden_size_per_layer_input", 0) or 0) > 0 for block in blocks
        )
        tied_embeddings = getattr(embedding_module, "weight", None) is getattr(head_module, "weight", None)
        return Gemma4Topology(
            text_backbone=text_backbone,
            blocks=blocks,
            embedding=ModuleTarget(name="model.language_model.embed_tokens", module=embedding_module),
            lm_head=ModuleTarget(name="lm_head", module=head_module),
            residual_writers=tuple(writers),
            multimodal_projections=tuple(multimodal),
            final_norm=ModuleTarget(name="model.language_model.norm", module=final_norm_module),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            vocab_size=vocab_size,
            moe_enabled=moe_enabled,
            ple_enabled=ple_enabled,
            tied_embeddings=tied_embeddings,
        )


__all__ = ["Gemma4Adapter", "Gemma4Topology"]
