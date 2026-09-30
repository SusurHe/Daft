from __future__ import annotations

import importlib.metadata as md
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from daft.storage.contracts import Layer, Location, LocationSource, ProviderInfo, ProviderKind
from daft.storage.errors import AmbiguousFormatError, ProviderNotFoundError

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: Entry point group third-party packages can use to register providers.
PROVIDER_ENTRY_POINT_GROUP = "daft.storage.providers"

#: Canonical format key -> file extensions used to infer the format from a path.
FORMAT_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "parquet": (".parquet", ".pq"),
    "csv": (".csv",),
    "json": (".json", ".jsonl", ".ndjson"),
    "avro": (".avro",),
    "text": (".txt", ".text"),
    "mcap": (".mcap",),
    "warc": (".warc",),
    "lance": (".lance",),
}

#: Schemes that denote the local filesystem. An empty scheme means a bare path.
LOCAL_SCHEMES: tuple[str, ...] = ("", "file", "local")

_REGISTRY: dict[tuple[ProviderKind, str], Any] = {}
_BY_NAME: dict[str, Any] = {}
_DISCOVERED = False
_BUILTINS_LOADED = False


def ensure_builtins() -> None:
    """Register the built-in providers exactly once.

    Called lazily from every lookup so that ``daft.storage`` works in a fresh interpreter without
    any explicit setup step, which mirrors how Daft registers its built-in functions once at module
    initialisation.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from daft.storage.providers import register_builtin_providers

    register_builtin_providers()


def register(provider: Any, *, override: bool = False) -> ProviderInfo:
    """Register a provider under every key it declares.

    Args:
        provider: Provider object exposing an ``info`` attribute.
        override: Allow replacing an existing registration for the same key.

    Returns:
        The registered provider's :class:`ProviderInfo`.

    Raises:
        ValueError: If a key is already taken and ``override`` is ``False``.
    """
    info: ProviderInfo = provider.info
    conflicts = [
        key for key in info.keys if (info.kind, key) in _REGISTRY and _REGISTRY[(info.kind, key)] is not provider
    ]
    if conflicts and not override:
        existing = {key: _REGISTRY[(info.kind, key)].info.name for key in conflicts}
        raise ValueError(
            f"Cannot register provider {info.name!r}: {info.kind.value} key(s) {conflicts} already registered by {existing}. "
            "Pass override=True to replace them explicitly."
        )
    for key in info.keys:
        _REGISTRY[(info.kind, key)] = provider
    _BY_NAME[info.name] = provider
    return info


def unregister(name: str) -> None:
    """Remove a provider by name, dropping all of its keys."""
    provider = _BY_NAME.pop(name, None)
    if provider is None:
        return
    for key in list(_REGISTRY):
        if _REGISTRY[key] is provider:
            del _REGISTRY[key]


def get(name: str) -> Any:
    """Return a registered provider by name."""
    ensure_builtins()
    if name not in _BY_NAME:
        raise ProviderNotFoundError("provider", name, _BY_NAME.keys(), hint=_dependency_hint(name))
    return _BY_NAME[name]


def registered(kind: ProviderKind | None = None) -> list[ProviderInfo]:
    """Return the info of every registered provider, optionally filtered by layer."""
    ensure_builtins()
    infos = {provider.info.name: provider.info for provider in _BY_NAME.values()}
    selected = [info for info in infos.values() if kind is None or info.kind is kind]
    return sorted(selected, key=lambda info: info.name)


def registered_keys(kind: ProviderKind) -> tuple[str, ...]:
    """Return every registered key of a given layer."""
    ensure_builtins()
    return tuple(sorted(key for (key_kind, key) in _REGISTRY if key_kind is kind))


def resolve(kind: ProviderKind, key: str) -> Any:
    """Resolve a provider by layer and key, raising an actionable error when missing."""
    ensure_builtins()
    if (kind, key) not in _REGISTRY:
        discover()
    provider = _REGISTRY.get((kind, key))
    if provider is None:
        raise ProviderNotFoundError(
            kind.value,
            key,
            registered_keys(kind),
            hint=_dependency_hint(key),
        )
    return provider


def resolve_filesystem(scheme: str) -> Any:
    """Resolve a filesystem provider for a URI scheme."""
    normalized = "file" if scheme in LOCAL_SCHEMES else scheme
    return resolve(ProviderKind.FILESYSTEM, normalized)


def resolve_format(format_name: str) -> Any:
    """Resolve a file format provider."""
    return resolve(ProviderKind.FILE_FORMAT, format_name.lower())


def resolve_catalog(key: str) -> Any:
    """Resolve a catalog provider."""
    return resolve(ProviderKind.CATALOG, key)


def _dependency_hint(key: str) -> str | None:
    """Build an installation hint for known optional dependencies."""
    hints = {
        "clickhouse": "Install ClickHouse support with: pip install 'daft[clickhouse]'",
        "iceberg": "Install Iceberg support with: pip install 'daft[iceberg]'",
        "delta": "Install Delta Lake support with: pip install 'daft[deltalake]'",
        "lance": "Install Lance support with: pip install 'daft[lance]'",
    }
    return hints.get(key)


def discover() -> None:
    """Load third-party providers declared through entry points.

    Discovery is best effort: a provider whose optional dependency is missing must not break
    ``import daft``, so failures are logged and skipped.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    try:
        entry_points = md.entry_points(group=PROVIDER_ENTRY_POINT_GROUP)
    except Exception as error:  # noqa: BLE001 - metadata backend unavailable, must not be fatal
        logger.warning("Failed to enumerate Daft storage provider entry points: %s", error)
        return
    for entry_point in entry_points:
        try:
            register(entry_point.load()())
        except Exception as error:  # noqa: BLE001 - third party code, must never be fatal
            logger.warning("Failed to load Daft storage provider %r: %s", entry_point.name, error)


def reset(*, load_builtins: bool = True) -> None:
    """Clear the registry and optionally restore the built-in providers.

    Intended for tests and for embedders that want a pristine registry.
    """
    global _DISCOVERED, _BUILTINS_LOADED
    _REGISTRY.clear()
    _BY_NAME.clear()
    _DISCOVERED = False
    # An explicit opt out must survive the lazy loading in ensure_builtins(), otherwise an empty
    # registry could never be observed.
    _BUILTINS_LOADED = not load_builtins
    if load_builtins:
        ensure_builtins()


def parse_uri(uri: str) -> tuple[str, str]:
    """Split a URI into ``(scheme, path)``.

    Bare paths, ``file://`` URIs and Windows drive letters all resolve to the local filesystem.
    """
    if len(uri) > 1 and uri[1] == ":" and uri[0].isalpha():
        # Windows drive letter such as "D:\\data\\file.parquet" is a path, not a scheme.
        return "file", uri
    parts = urlsplit(uri)
    scheme = parts.scheme.lower()
    if scheme == "":
        return "file", uri
    if scheme == "file":
        return "file", parts.path
    path = uri.split("://", 1)[1] if "://" in uri else uri
    return scheme, path


def infer_format(uri: str) -> str | None:
    """Infer the file format from a path's extension, or return ``None`` when ambiguous."""
    _, path = parse_uri(uri)
    lowered = path.lower()
    for format_name, extensions in FORMAT_EXTENSIONS.items():
        if lowered.endswith(extensions):
            return format_name
    return None


def known_formats() -> tuple[str, ...]:
    """Return every format key that can be inferred from an extension."""
    return tuple(sorted(FORMAT_EXTENSIONS))


@dataclass(frozen=True)
class ResolvedSource:
    """Result of resolving a URI into the layers that participate in reading it.

    Attributes:
        uri: The original URI.
        format_name: Canonical format key, or ``None`` for database backed backends.
        storage_key: Filesystem scheme that was resolved, or ``None`` for database backed backends.
        layers: Layers involved, bottom-up.
        location: Resolved location, or ``None`` when the backend owns its own files.
        location_source: Who supplied the location.
        direct_provider: Name of a non file provider (database or table format) owning the URI.
        trace: Human readable resolution log.
    """

    uri: str
    format_name: str | None
    storage_key: str | None
    layers: tuple[Layer, ...]
    location: Location | None
    location_source: LocationSource
    direct_provider: str | None = None
    trace: tuple[str, ...] = ()

    @property
    def storage(self) -> Any | None:
        """The filesystem provider, or ``None`` when no filesystem participates."""
        return resolve_filesystem(self.storage_key) if self.storage_key else None

    @property
    def format(self) -> Any | None:
        """The file format provider, or ``None`` when no file format participates."""
        return resolve_format(self.format_name) if self.format_name else None

    @property
    def direct(self) -> Any | None:
        """The database or table format provider that owns this URI, if any."""
        return get(self.direct_provider) if self.direct_provider else None


def resolve_uri(uri: str, *, format: str | None = None) -> ResolvedSource:
    """Resolve a URI into a storage provider, a file format provider and a location.

    Two axes are resolved independently, mirroring how Daft already reads files: the storage layer
    comes from the scheme and the format comes either from an explicit argument or from the file
    extension (``get_tabular_files_scan`` in ``daft/io/common.py`` takes the same two inputs,
    ``storage_config`` and ``file_format_config``).

    Args:
        uri: Path or URI to resolve.
        format: Explicit format name, which wins over extension inference.

    Returns:
        The resolved source description.

    Raises:
        ProviderNotFoundError: If the scheme has no filesystem provider.
        AmbiguousFormatError: If the format cannot be inferred and was not given explicitly.
    """
    ensure_builtins()
    scheme, path = parse_uri(uri)

    # Database backed and table format backends terminate at the catalog layer: there is no file
    # format to infer and no location to resolve, because the backend owns its own files.
    for kind, layer in ((ProviderKind.DATABASE, Layer.CATALOG), (ProviderKind.TABLE_FORMAT, Layer.TABLE_FORMAT)):
        provider = _REGISTRY.get((kind, scheme))
        if provider is not None:
            return ResolvedSource(
                uri=uri,
                format_name=None,
                storage_key=None,
                layers=(layer,),
                location=None,
                location_source=LocationSource.NONE,
                direct_provider=provider.info.name,
                trace=(f"catalog: {scheme!r} -> {provider.info.name} ({kind.value} backed, no file format)",),
            )

    storage_key = "file" if scheme in LOCAL_SCHEMES else scheme
    storage_provider = resolve_filesystem(storage_key)

    format_name = format.lower() if format is not None else infer_format(uri)
    if format_name is None:
        raise AmbiguousFormatError(uri, known_formats())
    format_provider = resolve_format(format_name)

    trace = [
        f"storage: {storage_key!r} -> {storage_provider.info.name}",
        f"format: {format_name!r} -> {format_provider.info.name}"
        + (" (explicit)" if format is not None else " (inferred from extension)"),
    ]
    location = Location(scheme=storage_key, path=path)
    return ResolvedSource(
        uri=uri,
        format_name=format_name,
        storage_key=storage_key,
        layers=(Layer.FILESYSTEM, Layer.FILE_FORMAT),
        location=location,
        location_source=LocationSource.USER,
        trace=tuple(trace),
    )


def known_schemes() -> tuple[str, ...]:
    """Return every registered filesystem scheme."""
    return registered_keys(ProviderKind.FILESYSTEM)


def describe_registry() -> str:
    """Render the registry contents, grouped by layer."""
    lines: list[str] = []
    for kind in ProviderKind:
        infos = registered(kind)
        if not infos:
            continue
        lines.append(f"{kind.value}:")
        for info in infos:
            keys = ", ".join(info.keys)
            protocol = f", write={info.write_protocol.value}" if info.write_protocol else ""
            lines.append(f"  {info.name} [{keys}] api={info.api_level.value}{protocol}")
    return "\n".join(lines) if lines else "no providers registered"


__all__ = [
    "FORMAT_EXTENSIONS",
    "LOCAL_SCHEMES",
    "PROVIDER_ENTRY_POINT_GROUP",
    "ResolvedSource",
    "describe_registry",
    "discover",
    "ensure_builtins",
    "get",
    "infer_format",
    "known_formats",
    "known_schemes",
    "parse_uri",
    "register",
    "registered",
    "registered_keys",
    "reset",
    "resolve",
    "resolve_catalog",
    "resolve_filesystem",
    "resolve_format",
    "resolve_uri",
    "unregister",
]


def _unused(_: Sequence[Any]) -> None:  # pragma: no cover - keeps the Sequence import meaningful
    """Placeholder kept out of the public surface."""
