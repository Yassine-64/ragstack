class RagstackError(Exception):
    """Base class for every error raised by ragstack."""


class DocumentNotFound(RagstackError):
    pass


class Forbidden(RagstackError):
    pass


class EmbeddingDimensionMismatch(RagstackError):
    def __init__(self, expected: int, got: int) -> None:
        super().__init__(f"embedding dimension mismatch: index has {expected}, embedder gave {got}")


class GenerationFailed(RagstackError):
    pass
