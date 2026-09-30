from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

from daft.storage.contracts import (
    ProviderInfo,
    ProviderKind,
    PythonDataSink,
    TableCapability,
    TypeMapping,
    UnsupportedOperationError,
    WriteProtocol,
)
from daft.storage.errors import OptionError
from daft.storage.negotiation import Residual
from daft.storage.options import Option, OptionsContract
from daft.storage.providers import type_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Kinds ClickHouse stores natively through the clickhouse-connect driver.
_NATIVE_KINDS = ("primitive",)

#: Kinds that have to be serialised before they can reach a column.
_SERIALIZE_KINDS = ("list", "struct", "map")

#: Read options forwarded verbatim to ``read_sql``.
_SCAN_FORWARDED_OPTIONS = ("partition_col", "num_partitions", "partition_bound_strategy", "infer_schema")

#: Connection keys that may come either from the URI or from explicit options.
_CONNECTION_KEYS = ("host", "port", "user", "password", "database")


def parse_target(uri: str) -> dict[str, Any]:
    """Parse connection parameters and the table reference out of a ClickHouse URI.

    Supported shapes:

    - ``clickhouse://host[:port]/database/table``
    - ``clickhouse://user:pass@host[:port]/database/table``
    - ``clickhouse:///database.table`` (credentials supplied through options instead)

    Args:
        uri: URI to parse.

    Returns:
        A mapping that may contain ``host``, ``port``, ``user``, ``password``, ``database`` and
        ``table``. Keys that the URI does not carry are absent.
    """
    parts = urlsplit(uri)
    netloc, path = parts.netloc, parts.path.strip("/")
    result: dict[str, Any] = {}

    if parts.hostname:
        result["host"] = parts.hostname
        if parts.port:
            result["port"] = parts.port
        if parts.username:
            result["user"] = unquote(parts.username)
        if parts.password:
            result["password"] = unquote(parts.password)
        if path:
            segments = path.split("/")
            if len(segments) >= 2:
                result["database"] = "/".join(segments[:-1])
            result["table"] = segments[-1]
    elif netloc or path:
        # No host: the URI addresses a table directly, e.g. clickhouse://analytics.events
        result["table"] = (netloc or path).replace("/", ".")

    return result


def qualify_table(target: Mapping[str, Any]) -> str | None:
    """Return ``database.table`` when a database was parsed, otherwise the bare table."""
    table = target.get("table")
    if table is None:
        return None
    database = target.get("database")
    return f"{database}.{table}" if database else str(table)


class ClickHouseScanSource:
    """Scan source for a ClickHouse table.

    Reads go through ``daft.read_sql``, which already pushes projections and predicates into the
    generated SQL (``SQLConnection.construct_sql_query``) and prefers the ConnectorX engine for
    ClickHouse. ``applies_operators`` stays ``False`` because this bridge does not re-inject
    pushdowns into the reader; the engine re-applies requested operators above the scan.
    """

    applies_operators = False

    def __init__(self, uri: str, options: Mapping[str, Any]) -> None:
        self.uri = uri
        self.options = dict(options)
        self.required_columns: tuple[str, ...] | None = None

    def query(self) -> str:
        """Return the SQL to execute."""
        query = self.options.get("query")
        if query:
            return str(query)
        table = qualify_table(parse_target(self.uri))
        if table is None:
            raise OptionError(
                "clickhouse",
                "no table to read",
                details=["The URI has no table path and the 'query' option was not provided"],
                suggestions=[
                    "Use daft.open('clickhouse://user:pass@host:8123/database/table', url=...)",
                    "Or daft.open('clickhouse:///database.table', url=...)",
                    "Or pass query='SELECT ...'",
                ],
            )
        return f"SELECT * FROM {table}"

    def read(self) -> Any:
        """Read the table through the SQL connector."""
        import daft

        kwargs = {key: self.options[key] for key in _SCAN_FORWARDED_OPTIONS if key in self.options}
        return daft.read_sql(self.query(), self.options["url"], **kwargs)

    def push_filters(self, filters: Sequence[Any]) -> Residual[Any]:
        """Accept predicates, since the SQL layer rewrites them into the query it sends."""
        return Residual.accept_all(filters)

    def push_projection(self, columns: Sequence[str]) -> None:
        """Record the required columns so the SQL layer can project only those."""
        self.required_columns = tuple(columns)


class ClickHouseProvider:
    """ClickHouse provider.

    A ClickHouse table is database backed, so resolution terminates at the catalog layer: there is no
    filesystem scheme and no file format involved, and the database owns its own files. Writes reuse
    the existing :class:`~daft.io.clickhouse.clickhouse_data_sink.ClickHouseDataSink`, so no second
    write implementation is introduced.
    """

    info = ProviderInfo(
        name="clickhouse",
        kind=ProviderKind.DATABASE,
        keys=("clickhouse", "ch"),
        capabilities=frozenset({TableCapability.BATCH_READ, TableCapability.BATCH_WRITE}),
        write_protocol=WriteProtocol.APPEND_ONLY,
        requires=("clickhouse_connect",),
        doc="ClickHouse tables: reads via read_sql, writes via the clickhouse-connect based sink.",
    )
    type_mapping: TypeMapping = type_mapping(
        native=_NATIVE_KINDS,
        serialize=_SERIALIZE_KINDS,
        non_primitive="str",
    )
    scan_options = OptionsContract(
        required=(Option("url", doc="Connection string, for example clickhouse://user:pass@host:8123/db"),),
        optional=(
            Option("query", type=str, doc="SQL query; defaults to SELECT * FROM <table in the URI>"),
            Option("partition_col", type=str, doc="Column used to split the read into partitions"),
            Option("num_partitions", type=int, doc="Number of partitions to read"),
            Option(
                "partition_bound_strategy",
                choices=("min-max", "percentile"),
                default="min-max",
                doc="How partition bounds are computed",
            ),
            Option("infer_schema", type=bool, default=True, doc="Infer the schema from a sample"),
        ),
    )
    sink_options = OptionsContract(
        optional=(
            Option("table", type=str, doc="Target table, which must already exist; defaults to the URI path"),
            Option("host", type=str, doc="Override the host parsed from the URI"),
            Option("port", type=int, doc="Override the port parsed from the URI"),
            Option("user", type=str, doc="Override the user parsed from the URI"),
            Option("password", type=str, doc="Override the password parsed from the URI"),
            Option("database", type=str, doc="Override the database parsed from the URI"),
            Option("client_kwargs", type=dict, doc="Extra arguments for the clickhouse-connect client"),
            Option("write_kwargs", type=dict, doc="Extra arguments for the insert call, e.g. server settings"),
        )
    )

    def scan(self, uri: str, options: Mapping[str, Any]) -> ClickHouseScanSource:
        """Return a scan source that reads through ``read_sql``."""
        return ClickHouseScanSource(uri, options)

    def sink(self, uri: str, options: Mapping[str, Any]) -> PythonDataSink:
        """Return a Python data sink wrapping the existing ClickHouse sink.

        The sink is constructed lazily so that a missing ``clickhouse-connect`` dependency produces an
        actionable error instead of an import failure at ``import daft`` time.
        """
        return PythonDataSink(sink=build_data_sink(uri, options), provider=self.info.name)


def build_data_sink(uri: str, options: Mapping[str, Any]) -> Any:
    """Build the ClickHouse data sink from a URI plus options.

    Options win over values parsed from the URI. Validation happens before the optional driver is
    imported, so a bad target reports a configuration error rather than an import error.

    Args:
        uri: Target URI such as ``clickhouse://user:pass@host:8123/database/table``.
        options: Validated provider options.

    Returns:
        A configured ``ClickHouseDataSink``.

    Raises:
        OptionError: If no target table or no host can be determined.
        UnsupportedOperationError: If the clickhouse-connect driver is unavailable.
    """
    target = parse_target(uri)
    table = options.get("table") or target.get("table")
    if table is None:
        raise OptionError(
            "clickhouse",
            "no target table",
            details=["Neither the 'table' option nor a table path in the URI was provided"],
            suggestions=[
                "Use daft.open('clickhouse://user:pass@host:8123/database/table').write(df)",
                "Or daft.open('clickhouse:///database.table', host=..., user=...).write(df)",
            ],
        )

    connection: dict[str, Any] = {}
    for key in _CONNECTION_KEYS:
        value = options.get(key) if options.get(key) is not None else target.get(key)
        if value is not None:
            connection[key] = value
    if "host" not in connection:
        raise OptionError(
            "clickhouse",
            "no host to connect to",
            details=["Provide the host in the URI, for example clickhouse://host:8123/database/table"],
            suggestions=["Or pass host=<hostname> explicitly"],
        )

    try:
        from daft.io.clickhouse.clickhouse_data_sink import ClickHouseDataSink
    except ImportError as error:
        raise UnsupportedOperationError(
            op="write",
            provider="clickhouse",
            reason=f"the clickhouse-connect driver is not available ({error})",
            alternatives=[
                "Install it with: pip install 'daft[clickhouse]'",
                "Fall back to df.write_sql(...) with a SQLAlchemy ClickHouse dialect",
            ],
        ) from error

    return ClickHouseDataSink(
        str(table),
        **connection,
        client_kwargs=options.get("client_kwargs"),
        write_kwargs=options.get("write_kwargs"),
    )


def read_clickhouse(
    table: str,
    url: str,
    *,
    filters: Sequence[Any] = (),
    columns: Sequence[str] | None = None,
    limit: int | None = None,
    **options: Any,
) -> Any:
    """Read a ClickHouse table as a DataFrame through the unified entry point.

    Args:
        table: Table reference such as ``database.table``.
        url: Connection string, for example ``clickhouse://user:pass@host:8123/db``.
        filters: Predicates to apply.
        columns: Columns to project, or ``None`` for all columns.
        limit: Maximum number of rows.
        **options: Additional provider options, validated by the provider contract.

    Returns:
        A ``daft.DataFrame``.
    """
    from daft.storage.handle import open_uri

    handle = open_uri(f"clickhouse://{table}", url=url, **options)
    return handle.read(filters=filters, columns=columns, limit=limit)


def write_clickhouse(df: Any, table: str, url: str, **options: Any) -> Any:
    """Write a DataFrame to a ClickHouse table through the unified entry point.

    Args:
        df: DataFrame to write.
        table: Target table, which must already exist.
        url: Connection string, for example ``clickhouse://user:pass@host:8123/db``.
        **options: Additional provider options, validated by the provider contract.

    Returns:
        The write metrics DataFrame produced by the sink.
    """
    from daft.storage.handle import open_uri

    return open_uri(url).write(df, table=table, **options)


__all__ = [
    "ClickHouseProvider",
    "ClickHouseScanSource",
    "build_data_sink",
    "parse_target",
    "qualify_table",
    "read_clickhouse",
    "write_clickhouse",
]
