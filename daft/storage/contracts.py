from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from daft.storage.errors import UnsupportedOperationError
from daft.storage.options import OptionsContract

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


class ProviderKind(enum.Enum):
    """Which layer of the four-layer stack a provider belongs to.

    The layers are independent: filesystem (L0), file format (L1), table format (L2) and catalog
    (L3). Only location-backed tables pull in L2 -> L1 -> L0; database-backed catalogs terminate at
    L3 because the database owns its own files.
    """

    FILESYSTEM = "filesystem"
    FILE_FORMAT = "file_format"
    TABLE_FORMAT = "table_format"
    CATALOG = "catalog"
    DATABASE = "database"


class Layer(enum.Enum):
    """Layers that participated in resolving a table or path, in bottom-up order."""

    FILESYSTEM = "filesystem"
    FILE_FORMAT = "file_format"
    TABLE_FORMAT = "table_format"
    CATALOG = "catalog"


class ApiLevel(enum.Enum):
    """Whether a provider implements the new contract or falls back to the legacy path.

    Mirrors ``TableCapability.V1_FALLBACK`` in Spark DataSource V2: a provider may declare that it
    only supports the pre-existing ``DataSource`` / ``ScanOperator`` / ``DataSink`` interfaces, in
    which case the engine routes to the legacy code path and records a plan hint instead of
    rejecting the provider.
    """

    V2 = "v2"
    V1_FALLBACK = "v1_fallback"


class LocationSource(enum.Enum):
    """Who supplied the storage location of a table.

    ``USER`` means the caller passed a path, ``CATALOG`` means the catalog or its warehouse
    configuration resolved it, and ``NONE`` means the backend has no notion of a location (for
    example a database-managed table).
    """

    USER = "user"
    CATALOG = "catalog"
    NONE = "none"


class TableCapability(enum.Enum):
    """Coarse, planning-level capabilities of a table.

    These are deliberately few: they only cover decisions the planner makes before any negotiation
    happens, plus fast-fail checks. Fine-grained abilities are expressed by implementing the
    ``Supports*`` mixins in ``daft.storage.negotiation`` instead of declaring booleans here.
    """

    BATCH_READ = "batch_read"
    BATCH_WRITE = "batch_write"
    NAMED_TABLES = "named_tables"
    ACCEPT_ANY_SCHEMA = "accept_any_schema"
    TRUNCATE = "truncate"


class DTypeSupport(enum.Enum):
    """How a backend handles one Daft data type."""

    NATIVE = "native"
    SERIALIZE = "serialize"
    REJECT = "reject"


class WriteProtocol(enum.Enum):
    """Commit semantics of a write path.

    Modelled on the writer/committer split in Flink's ``SinkV2``. Declaring the protocol lets the
    engine generate correct behaviour (retry policy, cleanup on failure, exactly-once claims)
    instead of leaving it implicit in each connector.
    """

    APPEND_ONLY = "append_only"
    ATOMIC_COMMIT = "atomic_commit"
    TWO_PHASE = "two_phase"
    IDEMPOTENT_UPSERT = "idempotent_upsert"

    @property
    def is_atomic(self) -> bool:
        """Whether the backend exposes a single atomic commit point."""
        return self in (WriteProtocol.ATOMIC_COMMIT, WriteProtocol.TWO_PHASE)

    @property
    def may_duplicate_on_retry(self) -> bool:
        """Whether a retried write can duplicate rows, which callers must be told about."""
        return self is WriteProtocol.APPEND_ONLY

    @property
    def supports_abort(self) -> bool:
        """Whether partially written data can be rolled back."""
        return self in (WriteProtocol.ATOMIC_COMMIT, WriteProtocol.TWO_PHASE)


class MetadataCost(enum.Enum):
    """Relative cost of reading a metadata column, used to decide whether it is worth adding."""

    FREE = "free"
    CHEAP = "cheap"
    EXPENSIVE = "expensive"


@dataclass(frozen=True)
class Location:
    """A resolved storage location.

    Attributes:
        scheme: Filesystem scheme, for example ``"file"``, ``"s3"`` or ``"gs"``.
        path: Path within that filesystem.
        options: Backend specific options that were used to resolve the location.
    """

    scheme: str
    path: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Render the location as a URI."""
        return f"{self.scheme}://{self.path}" if self.scheme else self.path


@dataclass(frozen=True)
class TableRef:
    """Immutable metadata for a resolved table or path.

    This object is intentionally metadata-only: negotiation state lives in the scan builder, which
    mirrors the ``Table`` / ``ScanBuilder`` split in Spark DataSource V2 and keeps ``TableRef``
    cacheable and serialisable.

    Attributes:
        schema: Table schema. Kept loosely typed so this layer does not have to import the engine.
        location: Resolved location, or ``None`` for backends without one.
        file_format: File format name such as ``"parquet"``, when the table is file based.
        table_protocol: Table format name such as ``"iceberg"``, when a table protocol is involved.
        layers: Layers that participated in resolving this table, bottom-up.
        location_source: Who supplied the location.
    """

    schema: Any = None
    location: Location | None = None
    file_format: str | None = None
    table_protocol: str | None = None
    layers: tuple[Layer, ...] = ()
    location_source: LocationSource = LocationSource.NONE

    @property
    def requires_filesystem(self) -> bool:
        """Whether this table needs a filesystem to read or write its data."""
        return Layer.FILESYSTEM in self.layers


@dataclass(frozen=True)
class MetadataColumn:
    """A column that is derived from the source rather than stored in the data files."""

    name: str
    dtype: Any
    readable: bool = True
    cost: MetadataCost = MetadataCost.FREE


@dataclass(frozen=True)
class TypeMapping:
    """Declarative mapping between Daft data types and a backend's type system.

    Attributes:
        support: Callable returning the support level for a Daft data type.
        to_backend: Optional Daft dtype -> backend type conversion.
        from_backend: Optional backend type -> Daft dtype conversion.
        non_primitive: How nested or multimodal columns are handled when they are serialised.
    """

    support: Callable[[Any], DTypeSupport]
    to_backend: Callable[[Any], Any] | None = None
    from_backend: Callable[[Any], Any] | None = None
    non_primitive: Literal["error", "str", "bytes"] = "error"

    def check(self, dtype: Any) -> DTypeSupport:
        """Return the support level for a single data type."""
        return self.support(dtype)

    def rejected(self, fields: Iterable[tuple[str, Any]]) -> list[tuple[str, Any]]:
        """Return the ``(name, dtype)`` pairs that this mapping rejects."""
        return [(name, dtype) for name, dtype in fields if self.check(dtype) is DTypeSupport.REJECT]

    def matrix(self, dtypes: Iterable[Any]) -> dict[str, str]:
        """Return a ``{dtype: support}`` matrix, used by documentation and diagnostics."""
        return {str(dtype): self.check(dtype).value for dtype in dtypes}


@dataclass(frozen=True)
class ProviderInfo:
    """Static description of a provider.

    Attributes:
        name: Unique provider name.
        kind: Layer this provider belongs to.
        keys: Lookup keys: schemes for filesystem providers, format names for file format
            providers, catalog type identifiers for catalogs.
        api_level: Whether the provider implements the new contract or the legacy fallback.
        capabilities: Coarse planning capabilities.
        write_protocol: Commit semantics of this provider's write path, if it can write.
        requires: Optional Python dependencies; used to build actionable import errors.
        doc: One line description.
    """

    name: str
    kind: ProviderKind
    keys: tuple[str, ...]
    api_level: ApiLevel = ApiLevel.V2
    capabilities: frozenset[TableCapability] = frozenset()
    write_protocol: WriteProtocol | None = None
    requires: tuple[str, ...] = ()
    doc: str = ""

    def can_read(self) -> bool:
        """Whether this provider advertises batch reads."""
        return TableCapability.BATCH_READ in self.capabilities

    def can_write(self) -> bool:
        """Whether this provider advertises batch writes."""
        return TableCapability.BATCH_WRITE in self.capabilities


@dataclass(frozen=True)
class NativeTabularSink:
    """Write via the engine's native writers (``LogicalPlanBuilder.write_tabular``)."""

    root_dir: str
    file_format: str
    write_mode: str = "append"
    format_options: Mapping[str, Any] = field(default_factory=dict)
    partition_cols: tuple[str, ...] = ()
    single_file: bool = False


@dataclass(frozen=True)
class CatalogSink:
    """Write through a table protocol commit (Iceberg, Delta, Paimon, Lance)."""

    table: Any
    table_format: str
    mode: str = "append"
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PythonDataSink:
    """Write through an existing ``daft.io.DataSink`` implementation dispatched by ``write_sink``."""

    sink: Any
    provider: str = ""


SinkSpec = NativeTabularSink | CatalogSink | PythonDataSink


def sink_form(spec: SinkSpec) -> Literal["native_tabular", "catalog", "python_datasink"]:
    """Return which of the three write forms a sink specification uses."""
    if isinstance(spec, NativeTabularSink):
        return "native_tabular"
    if isinstance(spec, CatalogSink):
        return "catalog"
    if isinstance(spec, PythonDataSink):
        return "python_datasink"
    raise TypeError(f"Unknown sink specification: {type(spec).__name__}")


@runtime_checkable
class SupportsScan(Protocol):
    """Provider that can produce a scan source for a URI."""

    def scan(self, uri: str, options: Mapping[str, Any]) -> Any:
        """Return a scan source (a ``DataSource``, ``ScanOperator`` or a V2 scan source)."""
        ...


@runtime_checkable
class SupportsSink(Protocol):
    """Provider that can describe a write for a URI."""

    def sink(self, uri: str, options: Mapping[str, Any]) -> SinkSpec:
        """Return the sink specification to use for this URI."""
        ...


@runtime_checkable
class SupportsCatalog(Protocol):
    """Provider that exposes named tables."""

    def catalog(self, uri: str, options: Mapping[str, Any]) -> Any:
        """Return a catalog object for this URI."""
        ...


@runtime_checkable
class Provider(Protocol):
    """The common contract every storage provider implements."""

    info: ProviderInfo
    scan_options: OptionsContract
    sink_options: OptionsContract


@runtime_checkable
class DataWriter(Protocol):
    """Worker-side writer, called for every micropartition, possibly in parallel.

    This is the first of the three write roles modelled on Flink's ``SinkV2``; it replaces the
    implicit "write and hope" behaviour of the current ``DataSink.write`` by producing an explicit
    committable that a committer can later commit or abort.
    """

    def write(self, micropartition: Any) -> Any:
        """Write one micropartition and return an optional committable."""
        ...

    def prepare_commit(self) -> Any:
        """Return the committable to hand to a committer."""
        ...

    def abort(self) -> None:
        """Discard anything written by this writer."""
        ...


@runtime_checkable
class Committer(Protocol):
    """Per-task or per-partition committer."""

    def commit(self, committables: Sequence[Any]) -> Sequence[Any]:
        """Commit committables produced by writers of this task."""
        ...


@runtime_checkable
class GlobalCommitter(Protocol):
    """Driver-side committer that merges committables across tasks."""

    def combine(self, committables: Sequence[Any]) -> Sequence[Any]:
        """Merge committables from all tasks into the final set."""
        ...

    def commit(self, committables: Sequence[Any]) -> None:
        """Perform the global commit."""
        ...

    def abort(self, committables: Sequence[Any]) -> None:
        """Roll back a failed global commit."""
        ...


def precheck_dtypes(
    provider: str,
    fields: Iterable[tuple[str, Any]],
    mapping: TypeMapping,
    alternatives: Sequence[str] = (),
) -> None:
    """Fail fast, before any data is written, when a column type is not writable.

    Args:
        provider: Provider name, used in the error message.
        fields: ``(column name, dtype)`` pairs of the data being written.
        mapping: The provider's type mapping.
        alternatives: Actionable alternatives shown to the user.

    Raises:
        UnsupportedOperationError: If at least one column would be rejected.
    """
    rejected = mapping.rejected(fields)
    if not rejected:
        return
    columns = ", ".join(f"{name}:{dtype}" for name, dtype in rejected)
    raise UnsupportedOperationError(
        op="write",
        provider=provider,
        reason=f"the following column types are not supported: {columns}",
        alternatives=list(alternatives)
        or [
            "Convert the column first, for example with image_encode(...) or cast to Binary",
            f"Check the full support matrix with daft.storage.dtype_matrix({provider!r})",
        ],
    )


__all__ = [
    "ApiLevel",
    "CatalogSink",
    "Committer",
    "DTypeSupport",
    "DataWriter",
    "GlobalCommitter",
    "Layer",
    "Location",
    "LocationSource",
    "MetadataColumn",
    "MetadataCost",
    "NativeTabularSink",
    "OptionsContract",
    "Provider",
    "ProviderInfo",
    "ProviderKind",
    "PythonDataSink",
    "SinkSpec",
    "SupportsCatalog",
    "SupportsScan",
    "SupportsSink",
    "TableCapability",
    "TableRef",
    "TypeMapping",
    "UnsupportedOperationError",
    "WriteProtocol",
    "precheck_dtypes",
    "sink_form",
]
