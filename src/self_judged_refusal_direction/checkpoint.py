from __future__ import annotations

import json
import os
import sqlite3
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from self_judged_refusal_direction.errors import ArtifactError
from self_judged_refusal_direction.hashing import bytes_sha256, canonical_json_bytes, object_sha256

_DATABASE_NAME = "checkpoint.sqlite3"
_IDENTITY_TABLE = "checkpoint_identity"
_ROWS_TABLE = "checkpoint_rows"


@dataclass(frozen=True)
class CheckpointEntry:
    ordinal: int
    prompt_key: str
    payload: dict[str, Any]
    payload_sha256: str


class PrivateCheckpoint:
    def __init__(
        self,
        directory: str | Path,
        *,
        identity: str,
        prompt_keys: Sequence[str],
    ):
        if type(identity) is not str or not identity:
            raise ArtifactError("checkpoint identity must be a non-empty string")
        keys = tuple(prompt_keys)
        if not keys or any(type(key) is not str or not key for key in keys):
            raise ArtifactError("checkpoint prompt keys must be non-empty strings")
        if len(set(keys)) != len(keys):
            raise ArtifactError("checkpoint prompt keys must be unique")
        self.identity = identity
        self.prompt_keys = keys
        self.directory = Path(directory).resolve()
        self.path = self.directory / _DATABASE_NAME
        self._connection: sqlite3.Connection | None = None
        self._open()

    def _open(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not self.directory.is_dir():
                raise ArtifactError(f"checkpoint directory is not a directory: {self.directory}")
            os.chmod(self.directory, 0o700)
            created = self._prepare_database_file()
            connection = sqlite3.connect(self.path)
            self._connection = connection
            os.chmod(self.path, 0o600)
            journal_mode = connection.execute("PRAGMA journal_mode = DELETE").fetchone()
            connection.execute("PRAGMA synchronous = FULL")
            synchronous = connection.execute("PRAGMA synchronous").fetchone()
            if journal_mode != ("delete",) or synchronous != (2,):
                raise ArtifactError("checkpoint durability settings are unavailable")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.execute("PRAGMA busy_timeout = 5000")
            self._initialize_schema()
            self._validate_schema()
            self._validate_identity()
            self._validate_permissions()
            if created:
                _fsync_directory(self.directory)
        except ArtifactError:
            self.close()
            raise
        except (OSError, sqlite3.Error) as error:
            self.close()
            raise ArtifactError(f"checkpoint could not be opened: {self.path}") from error

    def _prepare_database_file(self) -> bool:
        if self.path.exists():
            if self.path.is_symlink() or not self.path.is_file():
                raise ArtifactError(f"checkpoint database is not a regular file: {self.path}")
            os.chmod(self.path, 0o600)
            return False
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if self.path.is_symlink() or not self.path.is_file():
                raise ArtifactError(f"checkpoint database is not a regular file: {self.path}") from None
            os.chmod(self.path, 0o600)
            return False
        os.close(descriptor)
        return True

    def _initialize_schema(self) -> None:
        connection = self._require_connection()
        plan_sha256 = object_sha256(self.prompt_keys)
        try:
            connection.execute("BEGIN IMMEDIATE")
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            expected_tables = {_IDENTITY_TABLE, _ROWS_TABLE}
            if tables and tables != expected_tables:
                raise ArtifactError("checkpoint database has invalid tables")
            fresh = not tables
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_IDENTITY_TABLE} (
                    singleton INTEGER NOT NULL PRIMARY KEY CHECK (singleton = 1),
                    identity TEXT NOT NULL CHECK (length(identity) > 0),
                    prompt_plan_sha256 TEXT NOT NULL CHECK (length(prompt_plan_sha256) = 64),
                    prompt_count INTEGER NOT NULL CHECK (prompt_count > 0)
                ) STRICT
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_ROWS_TABLE} (
                    ordinal INTEGER NOT NULL PRIMARY KEY CHECK (ordinal >= 0),
                    prompt_key TEXT NOT NULL UNIQUE CHECK (length(prompt_key) > 0),
                    payload BLOB NOT NULL,
                    payload_sha256 TEXT NOT NULL CHECK (length(payload_sha256) = 64)
                ) STRICT
                """
            )
            identity_rows = connection.execute(f"SELECT singleton FROM {_IDENTITY_TABLE}").fetchall()
            if fresh:
                connection.execute(
                    f"""
                    INSERT INTO {_IDENTITY_TABLE}
                        (singleton, identity, prompt_plan_sha256, prompt_count)
                    VALUES (1, ?, ?, ?)
                    """,
                    (self.identity, plan_sha256, len(self.prompt_keys)),
                )
            elif len(identity_rows) != 1:
                raise ArtifactError("checkpoint database has invalid identity metadata")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _validate_schema(self) -> None:
        connection = self._require_connection()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if tables != {_IDENTITY_TABLE, _ROWS_TABLE}:
            raise ArtifactError("checkpoint database has invalid tables")
        expected_columns = {
            _IDENTITY_TABLE: (
                ("singleton", "INTEGER", 1, 1),
                ("identity", "TEXT", 1, 0),
                ("prompt_plan_sha256", "TEXT", 1, 0),
                ("prompt_count", "INTEGER", 1, 0),
            ),
            _ROWS_TABLE: (
                ("ordinal", "INTEGER", 1, 1),
                ("prompt_key", "TEXT", 1, 0),
                ("payload", "BLOB", 1, 0),
                ("payload_sha256", "TEXT", 1, 0),
            ),
        }
        for table, expected in expected_columns.items():
            observed = tuple(
                (row[1], row[2], row[3], row[5]) for row in connection.execute(f"PRAGMA table_info({table})")
            )
            if observed != expected:
                raise ArtifactError(f"checkpoint database has an invalid {table} schema")

    def _validate_identity(self) -> None:
        connection = self._require_connection()
        rows = connection.execute(
            f"SELECT identity, prompt_plan_sha256, prompt_count FROM {_IDENTITY_TABLE}"
        ).fetchall()
        if len(rows) != 1:
            raise ArtifactError("checkpoint database has invalid identity metadata")
        identity, prompt_plan_sha256, prompt_count = rows[0]
        if identity != self.identity:
            raise ArtifactError("checkpoint identity does not match")
        if prompt_plan_sha256 != object_sha256(self.prompt_keys) or prompt_count != len(self.prompt_keys):
            raise ArtifactError("checkpoint prompt plan does not match")

    def _validate_permissions(self) -> None:
        if stat.S_IMODE(self.directory.stat().st_mode) != 0o700:
            raise ArtifactError("checkpoint directory permissions are invalid")
        if stat.S_IMODE(self.path.stat().st_mode) != 0o600:
            raise ArtifactError("checkpoint database permissions are invalid")

    def load(self) -> tuple[CheckpointEntry, ...]:
        connection = self._require_connection()
        try:
            self._validate_schema()
            self._validate_identity()
            self._validate_permissions()
            integrity = connection.execute("PRAGMA integrity_check").fetchall()
            if integrity != [("ok",)]:
                raise ArtifactError("checkpoint database integrity check failed")
            rows = connection.execute(
                f"SELECT ordinal, prompt_key, payload, payload_sha256 FROM {_ROWS_TABLE} ORDER BY ordinal"
            ).fetchall()
            return tuple(self._entry_from_row(row) for row in rows)
        except ArtifactError:
            raise
        except (OSError, sqlite3.Error) as error:
            raise ArtifactError("checkpoint database could not be read") from error

    def _entry_from_row(self, row: tuple[Any, ...]) -> CheckpointEntry:
        if len(row) != 4:
            raise ArtifactError("checkpoint row has invalid fields")
        ordinal, prompt_key, payload_bytes, payload_sha256 = row
        if type(ordinal) is not int or not 0 <= ordinal < len(self.prompt_keys):
            raise ArtifactError("checkpoint row has an invalid ordinal")
        if type(prompt_key) is not str or prompt_key != self.prompt_keys[ordinal]:
            raise ArtifactError("checkpoint row has an invalid prompt key")
        if type(payload_bytes) is not bytes or not _valid_sha256(payload_sha256):
            raise ArtifactError("checkpoint row has invalid payload storage")
        if bytes_sha256(payload_bytes) != payload_sha256:
            raise ArtifactError("checkpoint row payload hash does not match")
        try:
            payload = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactError("checkpoint row payload is invalid JSON") from error
        if not isinstance(payload, dict) or canonical_json_bytes(payload) != payload_bytes:
            raise ArtifactError("checkpoint row payload is not a canonical object")
        return CheckpointEntry(
            ordinal=ordinal,
            prompt_key=prompt_key,
            payload=payload,
            payload_sha256=payload_sha256,
        )

    def write(
        self,
        ordinal: int,
        prompt_key: str,
        payload: Mapping[str, Any],
    ) -> CheckpointEntry:
        if type(ordinal) is not int or not 0 <= ordinal < len(self.prompt_keys):
            raise ArtifactError("checkpoint ordinal is outside the prompt plan")
        if type(prompt_key) is not str or prompt_key != self.prompt_keys[ordinal]:
            raise ArtifactError("checkpoint prompt key does not match its ordinal")
        payload_bytes, normalized = _encode_payload(payload)
        payload_sha256 = bytes_sha256(payload_bytes)
        entry = CheckpointEntry(
            ordinal=ordinal,
            prompt_key=prompt_key,
            payload=normalized,
            payload_sha256=payload_sha256,
        )
        connection = self._require_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            conflicts = connection.execute(
                f"""
                SELECT ordinal, prompt_key, payload, payload_sha256
                FROM {_ROWS_TABLE}
                WHERE ordinal = ? OR prompt_key = ?
                """,
                (ordinal, prompt_key),
            ).fetchall()
            if conflicts:
                if len(conflicts) != 1 or self._entry_from_row(conflicts[0]) != entry:
                    raise ArtifactError("checkpoint row conflicts with existing data")
                connection.commit()
                return entry
            connection.execute(
                f"""
                INSERT INTO {_ROWS_TABLE} (ordinal, prompt_key, payload, payload_sha256)
                VALUES (?, ?, ?, ?)
                """,
                (ordinal, prompt_key, payload_bytes, payload_sha256),
            )
            connection.commit()
            return entry
        except ArtifactError:
            connection.rollback()
            raise
        except (OSError, sqlite3.Error) as error:
            connection.rollback()
            raise ArtifactError("checkpoint row could not be committed") from error

    def require_complete(self) -> tuple[CheckpointEntry, ...]:
        entries = self.load()
        if len(entries) != len(self.prompt_keys):
            raise ArtifactError(f"checkpoint is incomplete: {len(entries)}/{len(self.prompt_keys)} rows")
        return entries

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ArtifactError("checkpoint database is closed")
        return self._connection

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _encode_payload(payload: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ArtifactError("checkpoint payload must be an object")
    try:
        payload_bytes = canonical_json_bytes(payload)
        normalized = json.loads(payload_bytes)
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError("checkpoint payload is not canonical JSON") from error
    if not isinstance(normalized, dict):
        raise ArtifactError("checkpoint payload must be an object")
    return payload_bytes, normalized


def _valid_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["CheckpointEntry", "PrivateCheckpoint"]
