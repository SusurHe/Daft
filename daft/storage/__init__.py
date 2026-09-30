from __future__ import annotations

"""Unified storage provider entry point.

This package provides the zero-breaking-change front door for every backend Daft talks to. The
design follows a four layer model in which the layers stay independent:

- L0 filesystem: schemes such as ``file``, ``s3``, ``gs``, ``az`` or ``hdfs``.
- L1 file format: ``parquet``, ``csv``, ``json``, ``avro``, ...
- L2 table format: ``iceberg``, ``delta``, ``paimon``, ...
- L3 catalog: named tables, which may be location backed (Hive, lakehouse) or database backed
  (Postgres, ClickHouse), the latter owning their own files.

Only location backed tables pull in L0/L1/L2, which is why ``TableRef`` carries an optional
``location`` plus a ``location_source`` saying who supplied it.

Capabilities come in two flavours, following Spark DataSource V2: coarse planning capabilities
(``TableCapability``) and fine grained abilities expressed by implementing the ``Supports*`` mixins
in :mod:`daft.storage.negotiation`. Pushdown is negotiated, never assumed: a source returns the
operators it did *not* absorb so the engine can re-evaluate them.
"""

from daft.storage.conformance import assert_conformance, check_provider, check_providers
from daft.storage.contracts import (
    ApiLevel,
    CatalogSink,
    Committer,
    DataWriter,
    DTypeSupport,
    GlobalCommitter,
    Layer,
    Location,
    LocationSource,
    MetadataColumn,
    MetadataCost,
    NativeTabularSink,
    Provider,
    ProviderInfo,
    ProviderKind,
    PythonDataSink,
    SinkSpec,
    TableCapability,
    TableRef,
    TypeMapping,
    UnsupportedOperationError,
    WriteProtocol,
    precheck_dtypes,
    sink_form,
)
from daft.storage.errors import (
    AmbiguousFormatError,
    OptionError,
    ProviderNotFoundError,
    StorageProviderError,
)
from daft.storage.handle import (
    StorageHandle,
    describe_uri,
    dtype_matrix,
    list_providers,
    open_uri,
)
from daft.storage.negotiation import (
    NEGOTIATION_ORDER,
    ScanPlan,
    ScanRequest,
    SupportsMetadataColumns,
    SupportsPushdownAggregates,
    SupportsPushdownFilters,
    SupportsPushdownLimit,
    SupportsPushdownProjection,
    SupportsReportPartitioning,
    SupportsReportStatistics,
    describe_source,
    negotiate,
)
from daft.storage.options import Option, OptionsContract
from daft.storage.registry import (
    FORMAT_EXTENSIONS,
    ResolvedSource,
    describe_registry,
    infer_format,
    parse_uri,
    register,
    registered,
    registered_keys,
    reset,
    resolve,
    resolve_uri,
    unregister,
)
from daft.storage.residual import Residual, accounts_for_all

__all__ = [
    "FORMAT_EXTENSIONS",
    "NEGOTIATION_ORDER",
    "AmbiguousFormatError",
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
    "Option",
    "OptionError",
    "OptionsContract",
    "Provider",
    "ProviderInfo",
    "ProviderKind",
    "ProviderNotFoundError",
    "PythonDataSink",
    "Residual",
    "ResolvedSource",
    "ScanPlan",
    "ScanRequest",
    "SinkSpec",
    "StorageHandle",
    "StorageProviderError",
    "SupportsMetadataColumns",
    "SupportsPushdownAggregates",
    "SupportsPushdownFilters",
    "SupportsPushdownLimit",
    "SupportsPushdownProjection",
    "SupportsReportPartitioning",
    "SupportsReportStatistics",
    "TableCapability",
    "TableRef",
    "TypeMapping",
    "UnsupportedOperationError",
    "WriteProtocol",
    "accounts_for_all",
    "assert_conformance",
    "check_provider",
    "check_providers",
    "describe_registry",
    "describe_source",
    "describe_uri",
    "dtype_matrix",
    "infer_format",
    "list_providers",
    "negotiate",
    "open_uri",
    "parse_uri",
    "precheck_dtypes",
    "register",
    "registered",
    "registered_keys",
    "reset",
    "resolve",
    "resolve_uri",
    "sink_form",
    "unregister",
]
