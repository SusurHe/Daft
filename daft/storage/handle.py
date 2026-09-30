from __future__ import annotations

import functools
import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from daft.storage.contracts import (
    ApiLevel,
    CatalogSink,
    NativeTabularSink,
    Provider,
    PythonDataSink,
    sink_form,
)
from daft.storage.errors import UnsupportedOperationError
from daft.storage.negotiation import ScanPlan, ScanRequest, describe_source, negotiate
from daft.storage.registry import ResolvedSource, registered, resolve_uri

if TYPE_CHECKING:
    from collections.abc import Sequence

    from daft.storage.contracts import ProviderInfo, SinkSpec


@dataclass
class StorageHandle:
    """Unified entry point for a resolved URI.

    A handle binds the four-layer resolution result (see :func:`daft.storage.resolve_uri`) together
    with the validated options, and exposes read, write and introspection in one place:

    .. code-block:: python

        handle = daft.storage.open_uri("s3://bucket/table.parquet")
        handle.capabilities  # coarse planning capabilities
        handle.write_protocol  # commit semantics of the write path
        df = handle.read(filters=[col("x") > 1], limit=10)
        df.write_sink(handle.sink(compression="zstd"))
        print(handle.describe())

    Attributes:
        resolved: Resolution result describing which layers participate.
        options: Validated scan options.
    """

    resolved: ResolvedSource
    options: dict[str, Any] = field(default_factory=dict)
    _plan: ScanPlan | None = None

    @property
    def uri(self) -> str:
        """The URI this handle was opened for."""
        return self.resolved.uri

    @property
    def format_name(self) -> str:
        """Canonical file format key."""
        return self.resolved.format_name

    @property
    def provider_info(self) -> ProviderInfo:
        """Info of the provider that owns scan or sink for this URI."""
        return self._operation_provider().info

    @property
    def api_level(self) -> ApiLevel:
        """Whether the owning provider implements the V2 contract or the legacy fallback."""
        return self.provider_info.api_level

    @property
    def capabilities(self) -> frozenset[Any]:
        """Coarse planning capabilities of the owning provider."""
        return self.provider_info.capabilities

    @property
    def write_protocol(self) -> Any:
        """Commit semantics declared by the owning provider, if it can write."""
        return self.provider_info.write_protocol

    @property
    def location(self) -> Any:
        """Resolved storage location."""
        return self.resolved.location

    def _candidate_providers(self) -> tuple[Any, ...]:
        """Return the providers that may own this URI, most specific first.

        Database backed URIs resolve to a single provider; file based URIs resolve to a format
        provider (scan and sink) plus a filesystem provider (credentials and transport).
        """
        return tuple(
            provider
            for provider in (self.resolved.direct, self.resolved.format, self.resolved.storage)
            if provider is not None
        )

    def _provider_for(self, protocol: type) -> Any | None:
        """Return the first candidate provider implementing a protocol."""
        for candidate in self._candidate_providers():
            if isinstance(candidate, protocol):
                return candidate
        return None

    def _operation_provider(self) -> Provider:
        """Return the provider that hosts scan and sink for this URI."""
        from daft.storage.contracts import SupportsScan, SupportsSink

        provider = self._provider_for((SupportsScan, SupportsSink))
        if provider is not None:
            return provider
        candidates = self._candidate_providers()
        if not candidates:  # pragma: no cover - resolution always yields at least one provider
            raise UnsupportedOperationError(
                op="resolve",
                provider=self.uri,
                reason="no provider was resolved for this URI",
                alternatives=["Check daft.storage.list_providers()"],
            )
        return candidates[0]

    def scan_source(self, **options: Any) -> Any:
        """Build the scan source, validating options against the provider contract."""
        from daft.storage.contracts import SupportsScan

        provider = self._provider_for(SupportsScan)
        if provider is None:
            raise UnsupportedOperationError(
                op="read",
                provider=self.provider_info.name,
                reason="this provider does not implement scan()",
                alternatives=[
                    "Use a read-only provider such as 'parquet' or 'csv'",
                    "Inspect registered providers with daft.storage.list_providers()",
                ],
            )
        merged = {**self.options, **options}
        validated = provider.scan_options.validate(merged, provider.info.name)
        return provider.scan(self.uri, validated)

    def plan(self, request: ScanRequest | None = None) -> ScanPlan:
        """Negotiate pushdowns without reading data.

        Args:
            request: Operators the engine would like pushed down.

        Returns:
            The negotiation result. Providers that only implement the legacy path return a plan that
            records the fallback instead of any pushdown.
        """
        source = self.scan_source()
        if self.api_level is ApiLevel.V1_FALLBACK:
            return ScanPlan(trace=[f"v1 fallback: {self.provider_info.name} uses the legacy scan path, no negotiation"])
        return negotiate(source, request or ScanRequest())

    def read(
        self,
        *,
        filters: Sequence[Any] = (),
        columns: Sequence[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        **options: Any,
    ) -> Any:
        """Read this URI as a DataFrame.

        The requested operators are negotiated with the scan source. Anything the source does not
        absorb is applied by the engine above the scan, so the result is identical whether or not a
        provider is able to push an operator down. Providers that do not actually apply operators
        themselves (``applies_operators = False``, which includes the legacy bridge) have every
        requested operator applied here.

        Args:
            filters: Predicates to apply.
            columns: Columns to project, or ``None`` for all columns.
            limit: Maximum number of rows.
            offset: Rows to skip before applying the limit.
            **options: Additional provider options, validated by the provider contract.

        Returns:
            A ``daft.DataFrame``.
        """
        from daft.storage.contracts import SupportsScan

        provider = self._provider_for(SupportsScan)
        if provider is None:  # pragma: no cover - guarded by the same check in scan_source()
            raise UnsupportedOperationError(
                op="read",
                provider=self.provider_info.name,
                reason="no candidate provider implements scan()",
                alternatives=["Inspect registered providers with daft.storage.list_providers()"],
            )
        merged = {**self.options, **options}
        validated = provider.scan_options.validate(merged, provider.info.name)
        source = provider.scan(self.uri, validated)

        request = ScanRequest(
            filters=tuple(filters),
            columns=tuple(columns) if columns is not None else None,
            limit=limit,
            offset=offset,
        )
        if self.api_level is ApiLevel.V1_FALLBACK:
            plan = ScanPlan(trace=[f"v1 fallback: {provider.info.name} uses the legacy scan path, no negotiation"])
        else:
            plan = negotiate(source, request)
        self._plan = plan

        if not hasattr(source, "read"):
            raise UnsupportedOperationError(
                op="read",
                provider=provider.info.name,
                reason="the scan source does not implement read()",
                alternatives=["Implement DataSource.read() on the source", "Use a built-in provider such as 'parquet'"],
            )
        df = source.read()
        return apply_operators(
            df, request, applies_operators=bool(getattr(source, "applies_operators", False)), plan=plan
        )

    def sink(self, **options: Any) -> SinkSpec:
        """Build a write specification for this URI.

        Args:
            **options: Additional provider options, validated by the provider contract.

        Returns:
            One of the three sink forms: native tabular writer, catalog commit, or a Python
            ``DataSink``. The form decides which existing engine path performs the write, so no new
            write channel is introduced.

        Raises:
            UnsupportedOperationError: If the provider cannot write.
        """
        from daft.storage.contracts import SupportsSink

        provider = self._provider_for(SupportsSink)
        if provider is None:
            raise UnsupportedOperationError(
                op="write",
                provider=self.provider_info.name,
                reason="this provider does not implement sink()",
                alternatives=[
                    "Use a writable provider such as 'parquet' or 'csv'",
                    "Fall back to an explicit writer, for example df.write_parquet(...)",
                ],
            )
        merged = {**self.options, **options}
        validated = provider.sink_options.validate(merged, provider.info.name)
        return provider.sink(self.uri, validated)

    def write(self, df: Any, **options: Any) -> Any:
        """Write a DataFrame through this handle.

        The sink form decides which existing engine path performs the write, so no new write channel
        is introduced: native tabular sinks go to ``write_parquet`` / ``write_csv``, catalog sinks go
        to the table protocol writer, and Python sinks go to ``write_sink``.

        Args:
            df: DataFrame to write.
            **options: Provider options, validated by the provider contract.

        Returns:
            The write result DataFrame returned by the underlying writer.
        """
        return write_with_spec(df, self.sink(**options))

    def describe(self) -> str:
        """Render a multi-line description of this handle for diagnostics.

        The output answers the questions that are hard to answer today: which layers participate,
        who supplied the location, what the provider can do, what the write commit semantics are, and
        what the last negotiation pushed down.
        """
        info = self.provider_info
        location = self.location if self.location is not None else "none"
        lines = [
            f"uri: {self.uri}",
            f"location: {location} (source={self.resolved.location_source.value})",
            f"layers: {[layer.value for layer in self.resolved.layers]}",
            f"provider: {info.name} (kind={info.kind.value}, api={info.api_level.value})",
            f"capabilities: {sorted(capability.value for capability in info.capabilities)}",
            f"write_protocol: {info.write_protocol.value if info.write_protocol else 'read-only'}",
        ]
        if info.requires:
            lines.append(f"requires: {list(info.requires)}")
        lines.extend(self.resolved.trace)
        if self._plan is not None:
            lines.append("last scan plan:")
            lines.extend(f"  {line}" for line in self._plan.describe().splitlines())
        return "\n".join(lines)


def write_with_spec(df: Any, spec: SinkSpec) -> Any:
    """Dispatch a sink specification to the existing writer it maps to.

    Args:
        df: DataFrame to write.
        spec: One of the three sink forms.

    Returns:
        The writer's result DataFrame.

    Raises:
        UnsupportedOperationError: If no writer is wired for the given sink form yet.
    """
    form = sink_form(spec)
    if isinstance(spec, NativeTabularSink):
        partition_cols = list(spec.partition_cols) or None
        if spec.file_format == "parquet":
            return df.write_parquet(
                spec.root_dir,
                compression=spec.format_options.get("compression", "snappy"),
                write_mode=spec.write_mode,
                partition_cols=partition_cols,
                single_file=spec.single_file,
            )
        if spec.file_format == "csv":
            return df.write_csv(spec.root_dir, write_mode=spec.write_mode, partition_cols=partition_cols)
        raise UnsupportedOperationError(
            op="write",
            provider=spec.file_format,
            reason=f"no native writer is wired for file format {spec.file_format!r}",
            alternatives=[
                "Use the dedicated writer, for example df.write_parquet(...) or df.write_json(...)",
                "Implement sink() on the provider so it returns a supported NativeTabularSink",
            ],
        )
    if isinstance(spec, CatalogSink):
        if spec.table_format == "iceberg":
            return df.write_iceberg(spec.table, mode=spec.mode)
        raise UnsupportedOperationError(
            op="write",
            provider=spec.table_format,
            reason="no catalog writer is wired for this table format yet",
            alternatives=[f"Call the dedicated writer for {spec.table_format} directly"],
        )
    if isinstance(spec, PythonDataSink):
        return df.write_sink(spec.sink)
    raise UnsupportedOperationError(
        op="write",
        provider="unknown",
        reason=f"unknown sink form {form!r}",
        alternatives=["Return one of NativeTabularSink, CatalogSink or PythonDataSink from Provider.sink()"],
    )


def apply_operators(df: Any, request: ScanRequest, *, applies_operators: bool, plan: ScanPlan) -> Any:
    """Apply the scan request operators to a DataFrame.

    Args:
        df: DataFrame returned by the scan source.
        request: Operators that were requested.
        applies_operators: Whether the source already applied the operators it accepted.
        plan: Negotiation result, used to decide which operators are still outstanding.

    Returns:
        The DataFrame with the outstanding operators applied.
    """
    if applies_operators:
        residual_filters = plan.filters_residual
        need_columns = request.columns if plan.columns is None else None
        need_limit = request.limit if plan.limit is None else None
        need_offset = request.offset if plan.limit is None else 0
    else:
        residual_filters = request.filters
        need_columns = request.columns
        need_limit = request.limit
        need_offset = request.offset

    if residual_filters:
        df = df.where(functools.reduce(operator.and_, residual_filters))
    if need_columns is not None:
        df = df.select(*need_columns)
    if need_offset:
        df = df.offset(need_offset)
    if need_limit is not None:
        df = df.limit(need_limit)
    return df


def open_uri(uri: str, *, format: str | None = None, **options: Any) -> StorageHandle:
    """Open a URI and return a :class:`StorageHandle`.

    Args:
        uri: Path or URI to open.
        format: Explicit format name, which wins over extension inference.
        **options: Provider options, validated by the provider contract.

    Returns:
        A handle exposing ``read()``, ``sink()``, ``plan()`` and ``describe()``.
    """
    resolved = resolve_uri(uri, format=format)
    return StorageHandle(resolved=resolved, options=dict(options))


def describe_uri(uri: str, *, format: str | None = None, **options: Any) -> str:
    """Return the description of a URI without reading it."""
    return open_uri(uri, format=format, **options).describe()


def list_providers(kind: Any | None = None) -> list[ProviderInfo]:
    """Return the registered providers, optionally filtered by layer."""
    return registered(kind)


def describe_source_capabilities(uri: str, *, format: str | None = None) -> str:
    """Return which capability mixins the scan source of a URI implements."""
    handle = open_uri(uri, format=format)
    return describe_source(handle.scan_source())


def dtype_matrix(provider_name: str, dtypes: Sequence[Any] | None = None) -> dict[str, str]:
    """Return the ``{dtype: support}`` matrix of a provider.

    Args:
        provider_name: Registered provider name, for example ``"parquet"``.
        dtypes: Data types to check. Defaults to a representative set built from ``daft.datatype``.

    Returns:
        Mapping from dtype string to one of ``native``, ``serialize`` or ``reject``.
    """
    from daft.storage.registry import get

    provider = get(provider_name)
    return provider.type_mapping.matrix(dtypes if dtypes is not None else default_dtype_sample())


def default_dtype_sample() -> list[Any]:
    """Build a representative list of Daft data types for capability matrices."""
    from daft.datatype import DataType

    return [
        DataType.int64(),
        DataType.float64(),
        DataType.string(),
        DataType.binary(),
        DataType.bool(),
        DataType.timestamp("us"),
        DataType.date(),
        DataType.list(DataType.int64()),
        DataType.struct({"a": DataType.int64()}),
        DataType.tensor(DataType.float32()),
        DataType.embedding(DataType.float32(), 8),
        DataType.image("RGB", 2, 2),
    ]


__all__ = [
    "StorageHandle",
    "apply_operators",
    "default_dtype_sample",
    "describe_source_capabilities",
    "describe_uri",
    "dtype_matrix",
    "list_providers",
    "open_uri",
    "sink_form",
    "write_with_spec",
]
