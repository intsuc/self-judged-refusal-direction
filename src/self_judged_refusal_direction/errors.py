from enum import StrEnum


class RefusalDirectionError(Exception):
    pass


class ConfigurationError(RefusalDirectionError):
    pass


class ArtifactError(RefusalDirectionError):
    pass


class CompatibilityError(RefusalDirectionError):
    pass


class InvariantError(RefusalDirectionError):
    pass


class TargetParseErrorCode(StrEnum):
    INVALID_MODE = "TARGET_PARSE_INVALID_MODE"
    INVALID_INPUT = "TARGET_PARSE_INVALID_INPUT"
    INVALID_GRAMMAR = "TARGET_PARSE_INVALID_GRAMMAR"
    THINKING_OPEN_MISSING = "TARGET_PARSE_THINKING_OPEN_MISSING"
    THINKING_CLOSE_MISSING = "TARGET_PARSE_THINKING_CLOSE_MISSING"
    THINKING_DELIMITER_IN_CONTENT = "TARGET_PARSE_THINKING_DELIMITER_IN_CONTENT"
    TERMINAL_MISSING = "TARGET_PARSE_TERMINAL_MISSING"
    TRAILING_TOKENS = "TARGET_PARSE_TRAILING_TOKENS"
    OFFICIAL_PARSER_REJECTED = "TARGET_PARSE_OFFICIAL_PARSER_REJECTED"
    OFFICIAL_RESPONSE_INVALID = "TARGET_PARSE_OFFICIAL_RESPONSE_INVALID"
    RESPONSE_FIELDS_INVALID = "TARGET_PARSE_RESPONSE_FIELDS_INVALID"
    UNEXPECTED_THINKING = "TARGET_PARSE_UNEXPECTED_THINKING"
    DECODE_FAILED = "TARGET_PARSE_DECODE_FAILED"
    BOUNDARY_MISMATCH = "TARGET_PARSE_BOUNDARY_MISMATCH"
    INTERNAL = "TARGET_PARSE_INTERNAL"


class TargetParseError(InvariantError):
    def __init__(self, code: TargetParseErrorCode, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(detail)


class PipelineError(RefusalDirectionError):
    pass
