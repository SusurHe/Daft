from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class StorageProviderError(Exception):
    """Base class for all errors raised by the storage provider layer."""


class ProviderNotFoundError(StorageProviderError):
    """Raised when no provider is registered for a requested key (scheme, format or catalog type)."""

    def __init__(self, kind: str, key: str, known: Sequence[str], hint: str | None = None) -> None:
        self.kind = kind
        self.key = key
        self.known = tuple(sorted(known))
        self.hint = hint
        message = f"No {kind} provider is registered for {key!r}. Registered {kind} providers: {list(self.known)}"
        if hint:
            message = f"{message}\n{hint}"
        super().__init__(message)


class OptionError(StorageProviderError):
    """Raised when provider options are missing, unknown, or ill-typed.

    The message is designed to be actionable: it lists the offending options, the accepted
    options, and (for typos) the closest known option name.
    """

    def __init__(
        self, provider: str, problem: str, details: Sequence[str] = (), suggestions: Sequence[str] = ()
    ) -> None:
        self.provider = provider
        self.problem = problem
        self.details = tuple(details)
        self.suggestions = tuple(suggestions)
        lines = [f"Invalid options for provider {provider!r}: {problem}"]
        lines.extend(f"  - {detail}" for detail in self.details)
        lines.extend(f"  {suggestion}" for suggestion in self.suggestions)
        super().__init__("\n".join(lines))


class UnsupportedOperationError(StorageProviderError):
    """Raised when a provider cannot perform a requested operation.

    Every instance carries a non-empty list of alternatives so that callers (CLI, Python API and
    plan hints) can surface the same actionable guidance.
    """

    def __init__(self, op: str, provider: str, reason: str, alternatives: Sequence[str]) -> None:
        if not alternatives:
            raise ValueError("UnsupportedOperationError requires at least one alternative")
        self.op = op
        self.provider = provider
        self.reason = reason
        self.alternatives = tuple(alternatives)
        lines = [f"{provider!r} does not support {op!r}: {reason}", "Alternatives:"]
        lines.extend(f"  {index}. {alternative}" for index, alternative in enumerate(self.alternatives, start=1))
        super().__init__("\n".join(lines))


class AmbiguousFormatError(StorageProviderError):
    """Raised when a URI does not identify a single file format and no explicit format was given."""

    def __init__(self, uri: str, known_formats: Sequence[str]) -> None:
        self.uri = uri
        self.known_formats = tuple(sorted(known_formats))
        super().__init__(
            f"Could not infer a file format for {uri!r}. "
            f"Pass format=<name> explicitly, or use one of the known extensions: {list(self.known_formats)}"
        )
