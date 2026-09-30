"""Bridge between the storage provider layer and the legacy scan configuration types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from daft.storage.contracts import UnsupportedOperationError
from daft.storage.registry import resolve_uri

if TYPE_CHECKING:
    from collections.abc import Sequence


def first_path(path: str | Sequence[str]) -> str:
    """Return the first path of a possibly multi path input.

    Args:
        path: A single path or a sequence of paths.

    Returns:
        The first path, used to resolve which storage and format providers participate.

    Raises:
        ValueError: If an empty sequence is provided.
    """
    if isinstance(path, str):
        return path
    paths = list(path)
    if not paths:
        raise ValueError("expected at least one path")
    return paths[0]


def tabular_scan_configs(
    path: str | Sequence[str],
    *,
    format: str,
    multithreaded_io: bool,
    io_config: Any = None,
    **options: Any,
) -> tuple[Any, Any]:
    """Return the legacy ``(FileFormatConfig, StorageConfig)`` pair for a tabular scan.

    The provider layer decides which filesystem and which format participate, the format provider
    builds the reader configuration from the options, and the filesystem provider contributes the IO
    configuration. Callers keep their existing signatures and behaviour; only the construction of the
    two configuration objects moves behind the provider boundary.

    Args:
        path: Path or list of paths being read.
        format: Canonical format key, for example ``"parquet"``.
        multithreaded_io: Whether the runner uses multithreaded IO.
        io_config: User supplied IO configuration, if any.
        **options: Format specific reader options.

    Returns:
        The reader configuration and the storage configuration, ready to hand to
        ``get_tabular_files_scan``.

    Raises:
        UnsupportedOperationError: If the resolved format provider cannot build a reader
            configuration.
    """
    from daft.daft import StorageConfig

    resolved = resolve_uri(first_path(path), format=format)
    provider = resolved.format if resolved.format is not None else resolved.storage
    build_config = getattr(provider, "legacy_file_format_config", None)
    if build_config is None:
        raise UnsupportedOperationError(
            op="read",
            provider=provider.info.name,
            reason=f"the provider cannot build a reader configuration for format {format!r}",
            alternatives=[
                "Use a provider that implements legacy_file_format_config()",
                "Inspect registered providers with daft.storage.list_providers()",
            ],
        )

    file_format_config = build_config(options)

    storage = resolved.storage
    contribute_io = getattr(storage, "io_config", None)
    resolved_io_config = contribute_io(io_config) if callable(contribute_io) else io_config
    storage_config = StorageConfig(multithreaded_io, resolved_io_config)
    return file_format_config, storage_config


__all__ = ["first_path", "tabular_scan_configs"]
