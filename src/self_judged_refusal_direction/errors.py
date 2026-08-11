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


class PipelineError(RefusalDirectionError):
    pass
