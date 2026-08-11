from __future__ import annotations

from collections.abc import Callable

from self_judged_refusal_direction.config import ModelConfig, ProjectConfig
from self_judged_refusal_direction.errors import ConfigurationError
from self_judged_refusal_direction.models.base import ArchitectureAdapter
from self_judged_refusal_direction.models.gemma4 import Gemma4Adapter

AdapterType = type[ArchitectureAdapter]
_ADAPTERS: dict[str, AdapterType] = {}


def register_adapter(
    name: str,
    adapter_type: AdapterType | None = None,
) -> AdapterType | Callable[[AdapterType], AdapterType]:
    key = name.strip().lower()
    if not key:
        raise ConfigurationError("adapter name must not be empty")

    def register(value: AdapterType) -> AdapterType:
        current = _ADAPTERS.get(key)
        if current is not None and current is not value:
            raise ConfigurationError(f"adapter is already registered: {key}")
        _ADAPTERS[key] = value
        return value

    if adapter_type is None:
        return register
    return register(adapter_type)


def registered_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def create_adapter(name: str) -> ArchitectureAdapter:
    key = name.strip().lower()
    try:
        adapter_type = _ADAPTERS[key]
    except KeyError as error:
        raise ConfigurationError(f"unknown architecture adapter: {name}") from error
    return adapter_type()


def adapter_for_config(config: ModelConfig | ProjectConfig) -> ArchitectureAdapter:
    model_config = config.model if isinstance(config, ProjectConfig) else config
    return create_adapter(model_config.adapter)


register_adapter("gemma4", Gemma4Adapter)
register_adapter("gemma4adapter", Gemma4Adapter)

__all__ = [
    "adapter_for_config",
    "create_adapter",
    "register_adapter",
    "registered_adapters",
]
