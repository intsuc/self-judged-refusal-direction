from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor, chunk_elements: int = 4 * 1024 * 1024) -> str:
    source = tensor.detach().reshape(-1)
    digest = hashlib.sha256()
    digest.update(str(source.dtype).encode())
    digest.update(canonical_json_bytes(tuple(tensor.shape)))
    for offset in range(0, source.numel(), chunk_elements):
        chunk = source[offset : offset + chunk_elements].to(device="cpu").contiguous()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def named_tensors_sha256(tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    seen: set[tuple[int, int, str]] = set()
    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        storage = tensor.untyped_storage()
        identity = (storage.data_ptr(), storage.nbytes(), str(tensor.device))
        if identity in seen:
            continue
        seen.add(identity)
        digest.update(name.encode())
        digest.update(tensor_sha256(tensor).encode())
    return digest.hexdigest()
