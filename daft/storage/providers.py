from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from daft.storage.contracts import (
    ApiLevel,
    CatalogSink,
    DTypeSupport,
    MetadataColumn,
    MetadataCost,
    NativeTabularSink,
    ProviderInfo,
    ProviderKind,
    TableCapability,
    TypeMapping,
    WriteProtocol,
)
from daft.storage.errors import UnsupportedOperationError
from daft.storage.negotiation import Residual
from daft.storage.options import Option, OptionsContract
from daft.storage.registry import register

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Data type kinds that every file based backend stores natively.
_PRIMITIVE_KINDS = ("primitive",)

#: Nested kinds, used to build honest per format matrices.
_NESTED_KINDS = ("list", "struct", "map")

#: Multimodal kinds. Parquet round-trips them through Arrow extension metadata; CSV cannot.
_MULTIMODAL_KINDS = ("tensor", "sparse_tensor", "image", "embedding", "file", "python")


def dtype_kind(dtype: Any) -> str:
    """Classify a Daft data type into a coarse kind, using duck typing.

    Kept dependency free so the provider layer can be reasoned about (and unit tested) without the
    engine runtime. Real ``daft.datatype.DataType`` values and lightweight stand-ins both work.

    Args:
        dtype: A Daft data type, or any object exposing the ``is_*`` predicate methods.

    Returns:
        One of ``python``, ``tensor``, ``sparse_tensor``, ``image``, ``embedding``, ``file``,
        ``list``, ``struct``, ``map``, ``extension`` or ``primitive``.
    """
    checks = (
        ("python", "is_python"),
        ("tensor", "is_tensor"),
        ("tensor", "is_fixed_shape_tensor"),
        ("sparse_tensor", "is_sparse_tensor"),
        ("sparse_tensor", "is_fixed_shape_sparse_tensor"),
        ("image", "is_image"),
        ("image", "is_fixed_shape_image"),
        ("embedding", "is_embedding"),
        ("file", "is_file"),
        ("list", "is_list"),
        ("list", "is_fixed_size_list"),
        ("struct", "is_struct"),
        ("map", "is_map"),
        ("extension", "is_extension"),
    )
    for kind, predicate in checks:
        method = getattr(dtype, predicate, None)
        if callable(method) and method():
            return kind
    return "primitive"


def type_mapping(
    *,
    native: Sequence[str] = _PRIMITIVE_KINDS,
    serialize: Sequence[str] = (),
    non_primitive: str = "error",
) -> TypeMapping:
    """Build a declarative type mapping from data type kinds.

    Args:
        native: Kinds stored without conversion.
        serialize: Kinds that can be written after serialisation, for example JSON text.
        non_primitive: Strategy for serialised columns, mirroring ``write_sql`` semantics.

    Returns:
        A :class:`TypeMapping` in which every unlisted kind is rejected.
    """
    native_set = set(native)
    serialize_set = set(serialize)

    def support(dtype: Any) -> DTypeSupport:
        kind = dtype_kind(dtype)
        if kind in native_set:
            return DTypeSupport.NATIVE
        if kind in serialize_set:
            return DTypeSupport.SERIALIZE
        return DTypeSupport.REJECT

    return TypeMapping(support=support, non_primitive=non_primitive)  # type: ignore[arg-type]


class _FileScanSource:
    """Shared behaviour for file based V2 scan sources.

    A source declares which operators it can absorb through the capability mixins, but delegates the
    actual read to the existing reader entry point. ``applies_operators`` is therefore ``False``: the
    engine re-applies every requested operator above the scan, which keeps user visible results
    identical to today's behaviour while the negotiation records what a native implementation would
    push down.
    """

    applies_operators = False

    def __init__(self, uri: str, options: Mapping[str, Any]) -> None:
        self.uri = uri
        self.options = dict(options)
        self.required_columns: tuple[str, ...] | None = None

    def read(self) -> Any:
        """Read the URI through the existing file reader entry point."""
        raise NotImplementedError


class ParquetScanSource(_FileScanSource):
    """Parquet scan source.

    Absorbs predicates and column pruning because the Parquet reader prunes row groups from
    statistics and turns the evaluated mask into a ``RowSelection``. Limits are deliberately *not*
    absorbed, matching ``GlobScanOperator::can_absorb_limit`` returning ``false``.
    """

    def read(self) -> Any:
        """Read the parquet file or dataset through ``daft.read_parquet``."""
        import daft

        options = {key: value for key, value in self.options.items() if key != "infer_schema"}
        return daft.read_parquet(self.uri, **options)

    def push_filters(self, filters: Sequence[Any]) -> Residual[Any]:
        """Absorb every predicate, since the reader evaluates them while decoding."""
        return Residual.accept_all(filters)

    def push_projection(self, columns: Sequence[str]) -> None:
        """Record the required columns so the reader can prune the rest."""
        self.required_columns = tuple(columns)

    def metadata_columns(self) -> Sequence[MetadataColumn]:
        """Expose the file path column when one was requested."""
        name = self.options.get("file_path_column")
        if not name:
            return ()
        return (MetadataColumn(name=name, dtype="Utf8", cost=MetadataCost.FREE),)


class CsvScanSource(_FileScanSource):
    """CSV scan source.

    Only column pruning is absorbed. CSV has no per row statistics to skip on and Daft does not
    evaluate pushed predicates inside the CSV reader, so predicates and limits stay in the plan. This
    is the intended contrast with :class:`ParquetScanSource`: declared abilities match reality.
    """

    def read(self) -> Any:
        """Read the CSV file or dataset through ``daft.read_csv``."""
        import daft

        options = {key: value for key, value in self.options.items() if key != "infer_schema"}
        return daft.read_csv(self.uri, **options)

    def push_projection(self, columns: Sequence[str]) -> None:
        """Record the required columns so the reader can prune the rest."""
        self.required_columns = tuple(columns)


class LocalFileSystemProvider:
    """Filesystem provider for local paths. Transport only: it hosts neither scan nor sink."""

    info = ProviderInfo(
        name="local",
        kind=ProviderKind.FILESYSTEM,
        keys=("file", "local"),
        capabilities=frozenset(),
        doc="Local filesystem reader and writer.",
    )
    scan_options = OptionsContract(optional=(Option("io_config", doc="IO configuration for credentials"),))
    sink_options = OptionsContract(optional=(Option("io_config", doc="IO configuration for credentials"),))

    def io_config(self, io_config: Any = None) -> Any:
        """Return the IO configuration to use for this scheme.

        Local paths accept the user supplied configuration unchanged; object store providers will
        contribute defaults here once per-scheme configuration moves into the provider layer.
        """
        return io_config


class RemoteFileSystemProvider:
    """Filesystem provider for object stores served by an existing Daft IO backend."""

    scan_options = OptionsContract(optional=(Option("io_config", doc="IO configuration for credentials"),))
    sink_options = OptionsContract(optional=(Option("io_config", doc="IO configuration for credentials"),))

    def __init__(self, name: str, keys: Sequence[str], doc: str) -> None:
        self.info = ProviderInfo(
            name=name,
            kind=ProviderKind.FILESYSTEM,
            keys=tuple(keys),
            capabilities=frozenset(),
            doc=doc,
        )

    def io_config(self, io_config: Any = None) -> Any:
        """Return the IO configuration to use for this scheme.

        The credentials model is unchanged: whatever the user passed through ``io_config`` is used,
        and per-scheme defaults can be layered in here later without touching the readers.
        """
        return io_config


class ParquetFormatProvider:
    """Parquet provider: predicate and projection pushdown, atomic file commit."""

    info = ProviderInfo(
        name="parquet",
        kind=ProviderKind.FILE_FORMAT,
        keys=("parquet", "pq"),
        capabilities=frozenset({TableCapability.BATCH_READ, TableCapability.BATCH_WRITE}),
        write_protocol=WriteProtocol.ATOMIC_COMMIT,
        doc="Apache Parquet read through the arrow-rs based reader with row group pruning.",
    )
    type_mapping = type_mapping(
        native=(*_PRIMITIVE_KINDS, "extension", *_NESTED_KINDS, *_MULTIMODAL_KINDS),
        non_primitive="error",
    )
    scan_options = OptionsContract(
        optional=(
            Option("io_config", doc="IO configuration for remote storage"),
            Option("file_path_column", type=str, doc="Name of a generated file path column"),
            Option("hive_partitioning", type=bool, default=False, doc="Parse hive style partition columns"),
            Option("infer_schema", type=bool, default=True, doc="Infer the schema from the first file"),
        )
    )
    sink_options = OptionsContract(
        optional=(
            Option(
                "write_mode",
                choices=("append", "overwrite", "overwrite-partitions"),
                default="append",
                doc="Write mode",
            ),
            Option("compression", type=str, default="snappy", doc="Compression codec"),
            Option("partition_cols", type=tuple, doc="Columns to partition the output by"),
            Option("single_file", type=bool, default=False, doc="Write one file, native runner only"),
            Option("io_config", doc="IO configuration for remote storage"),
        )
    )

    def legacy_file_format_config(self, options: Mapping[str, Any]) -> Any:
        """Build the reader configuration for the existing file scan path.

        Keeping this mapping in the provider gives the format a single source of truth: the reader
        entry point only forwards its arguments, and both the capability declaration and the reader
        configuration live next to each other.
        """
        from daft.daft import FileFormatConfig, ParquetSourceConfig

        return FileFormatConfig.from_parquet_config(
            ParquetSourceConfig(
                coerce_int96_timestamp_unit=options.get("coerce_int96_timestamp_unit"),
                row_groups=options.get("row_groups"),
                chunk_size=options.get("chunk_size"),
                ignore_corrupt_files=bool(options.get("ignore_corrupt_files", False)),
            )
        )

    def scan(self, uri: str, options: Mapping[str, Any]) -> ParquetScanSource:
        """Return a Parquet scan source."""
        return ParquetScanSource(uri, options)

    def sink(self, uri: str, options: Mapping[str, Any]) -> NativeTabularSink:
        """Return a native tabular sink, keeping the fastest existing write path."""
        return NativeTabularSink(
            root_dir=uri,
            file_format="parquet",
            write_mode=options.get("write_mode", "append"),
            format_options={"compression": options.get("compression", "snappy")},
            partition_cols=tuple(options.get("partition_cols") or ()),
            single_file=bool(options.get("single_file", False)),
        )


class CsvFormatProvider:
    """CSV provider: projection pushdown only, atomic file commit, flat types only."""

    info = ProviderInfo(
        name="csv",
        kind=ProviderKind.FILE_FORMAT,
        keys=("csv",),
        capabilities=frozenset({TableCapability.BATCH_READ, TableCapability.BATCH_WRITE}),
        write_protocol=WriteProtocol.ATOMIC_COMMIT,
        doc="Comma separated values with parallel parse within a file and no predicate pushdown.",
    )
    type_mapping = type_mapping(native=_PRIMITIVE_KINDS, serialize=(), non_primitive="error")
    scan_options = OptionsContract(
        optional=(
            Option("io_config", doc="IO configuration for remote storage"),
            Option("infer_schema", type=bool, default=True, doc="Infer the schema from a sample"),
        )
    )
    sink_options = OptionsContract(
        optional=(
            Option(
                "write_mode",
                choices=("append", "overwrite", "overwrite-partitions"),
                default="append",
                doc="Write mode",
            ),
            Option("partition_cols", type=tuple, doc="Columns to partition the output by"),
            Option("io_config", doc="IO configuration for remote storage"),
        )
    )

    def scan(self, uri: str, options: Mapping[str, Any]) -> CsvScanSource:
        """Return a CSV scan source."""
        return CsvScanSource(uri, options)

    def sink(self, uri: str, options: Mapping[str, Any]) -> NativeTabularSink:
        """Return a native tabular sink."""
        return NativeTabularSink(
            root_dir=uri,
            file_format="csv",
            write_mode=options.get("write_mode", "append"),
            partition_cols=tuple(options.get("partition_cols") or ()),
        )


@dataclass
class LegacyDataSourceProvider:
    """Adapter exposing a legacy ``DataSource`` through the unified entry point.

    Registered as ``api_level=V1_FALLBACK``: the engine skips negotiation and records a plan hint
    instead of rejecting the provider. This mirrors ``TableCapability.V1_FALLBACK`` in Spark
    DataSource V2 and is what lets third party sources migrate incrementally.
    """

    info: ProviderInfo = field(
        default_factory=lambda: ProviderInfo(
            name="legacy-datasource",
            kind=ProviderKind.FILE_FORMAT,
            keys=("datasource",),
            api_level=ApiLevel.V1_FALLBACK,
            capabilities=frozenset({TableCapability.BATCH_READ}),
            doc="Wraps a user supplied DataSource using the legacy scan path.",
        )
    )
    scan_options: ClassVar[OptionsContract] = OptionsContract(
        required=(Option("source", doc="A DataSource instance, or a zero argument factory returning one"),)
    )
    sink_options: ClassVar[OptionsContract] = OptionsContract()

    def scan(self, uri: str, options: Mapping[str, Any]) -> Any:
        """Return the user supplied legacy source."""
        source = options["source"]
        return source() if callable(source) else source

    def sink(self, uri: str, options: Mapping[str, Any]) -> CatalogSink:
        """Legacy sources are read only in this skeleton."""
        raise UnsupportedOperationError(
            op="write",
            provider=self.info.name,
            reason="the legacy DataSource adapter is read only",
            alternatives=[
                "Implement a DataSink and call df.write_sink(...)",
                "Use a built-in writable provider such as 'parquet'",
            ],
        )


def _register_database_providers() -> None:
    """Register database backed providers.

    Their modules avoid optional imports at import time, so registration never requires the database
    driver to be installed; a missing driver surfaces as an actionable error when the provider is
    actually used.
    """
    from daft.io.clickhouse.provider import ClickHouseProvider

    register(ClickHouseProvider(), override=True)


def register_builtin_providers() -> None:
    """Register every built-in provider.

    Called by ``daft.storage.reset(load_builtins=True)`` and lazily by the registry on first use.
    """
    register(LocalFileSystemProvider(), override=True)
    register(RemoteFileSystemProvider("s3", ("s3", "s3a"), "Amazon S3 and S3 compatible object stores."), override=True)
    register(RemoteFileSystemProvider("gcs", ("gs", "gcs"), "Google Cloud Storage."), override=True)
    register(RemoteFileSystemProvider("azure", ("az", "abfs", "abfss"), "Azure Blob Storage and ADLS."), override=True)
    register(RemoteFileSystemProvider("http", ("http", "https"), "HTTP(S) endpoints with range reads."), override=True)
    register(RemoteFileSystemProvider("hf", ("hf",), "Hugging Face Hub datasets."), override=True)
    register(ParquetFormatProvider(), override=True)
    register(CsvFormatProvider(), override=True)
    register(LegacyDataSourceProvider(), override=True)
    _register_database_providers()
