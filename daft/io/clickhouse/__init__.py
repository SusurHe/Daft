from __future__ import annotations

from typing import TYPE_CHECKING

from daft.io.clickhouse.provider import ClickHouseProvider, read_clickhouse, write_clickhouse

if TYPE_CHECKING:
    from daft.io.clickhouse.clickhouse_data_sink import ClickHouseDataSink

__all__ = ["ClickHouseDataSink", "ClickHouseProvider", "read_clickhouse", "write_clickhouse"]


def __getattr__(name: str) -> object:
    """Import the data sink lazily so that a missing driver does not break `import daft`."""
    if name == "ClickHouseDataSink":
        from daft.io.clickhouse.clickhouse_data_sink import ClickHouseDataSink

        return ClickHouseDataSink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
