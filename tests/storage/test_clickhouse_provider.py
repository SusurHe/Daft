from __future__ import annotations

import sys

import pytest

from daft.io.clickhouse.provider import ClickHouseProvider, ClickHouseScanSource
from daft.storage import (
    DTypeSupport,
    Layer,
    LocationSource,
    OptionError,
    ProviderKind,
    PythonDataSink,
    ScanRequest,
    TableCapability,
    UnsupportedOperationError,
    WriteProtocol,
    dtype_matrix,
    negotiate,
    open_uri,
    precheck_dtypes,
    reset,
    resolve_uri,
    sink_form,
)

URI = "clickhouse://user:pass@host:8123/analytics/events"
URL = "clickhouse://user:pass@host:8123/analytics"


class FakeTensor:
    """Minimal stand-in for a multimodal Daft data type."""

    def is_tensor(self) -> bool:
        """Report as a tensor type."""
        return True

    def __str__(self) -> str:
        """Render a readable name for matrix output."""
        return "Tensor"


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Keep the provider registry deterministic across tests."""
    reset()
    yield
    reset()


def test_database_uri_terminates_at_the_catalog_layer():
    resolved = resolve_uri(URI)
    assert resolved.direct_provider == "clickhouse"
    assert resolved.layers == (Layer.CATALOG,)
    assert resolved.location is None
    assert resolved.location_source is LocationSource.NONE
    assert resolved.format_name is None
    assert resolved.storage_key is None
    assert resolved.storage is None
    assert resolved.format is None
    assert "no file format" in resolved.trace[0]


def test_handle_resolves_without_a_file_format():
    handle = open_uri(URI)
    assert handle.provider_info.name == "clickhouse"
    assert handle.location is None
    assert handle.write_protocol is WriteProtocol.APPEND_ONLY


def test_provider_declarations_are_honest():
    info = ClickHouseProvider().info
    assert info.kind is ProviderKind.DATABASE
    assert info.can_read() and info.can_write()
    assert info.write_protocol is WriteProtocol.APPEND_ONLY
    assert TableCapability.NAMED_TABLES not in info.capabilities, "a forwarding catalog is not implemented yet"
    assert "clickhouse_connect" in info.requires


def test_append_only_semantics_are_explicit():
    protocol = ClickHouseProvider().info.write_protocol
    assert not protocol.is_atomic
    assert protocol.may_duplicate_on_retry
    assert not protocol.supports_abort


def test_type_mapping_rejects_multimodal_columns_with_alternatives():
    mapping = ClickHouseProvider().type_mapping
    assert mapping.check(FakeTensor()) is DTypeSupport.REJECT
    with pytest.raises(UnsupportedOperationError) as error:
        precheck_dtypes("clickhouse", [("img", FakeTensor())], mapping)
    message = str(error.value)
    assert "img:Tensor" in message
    assert "image_encode" in message
    assert "dtype_matrix('clickhouse')" in message


def test_scan_negotiation_pushes_filters_and_projection_but_not_limits():
    source = ClickHouseScanSource(URI, {"url": URL})
    plan = negotiate(source, ScanRequest(filters=("a > 1",), columns=("a",), limit=5))
    assert plan.filters == ("a > 1",)
    assert not plan.filters_residual
    assert source.required_columns == ("a",)
    assert plan.limit is None, "the SQL reader has no limit pushdown, so the limit stays in the plan"


def test_scan_query_defaults_to_the_table_in_the_uri():
    assert ClickHouseScanSource(URI, {"url": URL}).query() == "SELECT * FROM analytics.events"
    assert ClickHouseScanSource("clickhouse:///db.events", {"url": URL}).query() == "SELECT * FROM db.events"
    assert ClickHouseScanSource(URI, {"url": URL, "query": "SELECT 1"}).query() == "SELECT 1"


def test_scan_requires_a_connection_string():
    with pytest.raises(OptionError, match="missing required option"):
        open_uri(URI).scan_source()


def test_sink_requires_a_target_table_before_touching_the_driver():
    with pytest.raises(OptionError, match="no target table"):
        ClickHouseProvider().sink("clickhouse://", {})


def test_sink_typo_gets_a_suggestion():
    with pytest.raises(OptionError) as error:
        ClickHouseProvider().sink_options.validate({"tabel": "t"}, "clickhouse")
    assert "Did you mean 'table'" in str(error.value)


def test_sink_returns_a_python_datasink_when_the_driver_is_present():
    pytest.importorskip("clickhouse_connect")
    spec = ClickHouseProvider().sink(URI, {})
    assert isinstance(spec, PythonDataSink)
    assert sink_form(spec) == "python_datasink"
    assert spec.provider == "clickhouse"
    # credentials, port and database are parsed out of the URI, table included
    sink = spec.sink
    assert sink._table == "events", "the sink receives the bare table name plus the database"
    assert sink._client_kwargs["host"] == "host"
    assert sink._client_kwargs["port"] == 8123
    assert sink._client_kwargs["user"] == "user"
    assert sink._client_kwargs["database"] == "analytics"


def test_sink_options_override_values_parsed_from_the_uri():
    pytest.importorskip("clickhouse_connect")
    spec = ClickHouseProvider().sink(URI, {"table": "other", "port": 9000})
    assert spec.sink._table == "other"
    assert spec.sink._client_kwargs["port"] == 9000


def test_sink_without_a_host_reports_a_configuration_error():
    with pytest.raises(OptionError, match="no host to connect to"):
        ClickHouseProvider().sink("clickhouse:///db.events", {})


def test_missing_driver_produces_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "clickhouse_connect", None)
    monkeypatch.delitem(sys.modules, "daft.io.clickhouse.clickhouse_data_sink", raising=False)
    with pytest.raises(UnsupportedOperationError) as error:
        ClickHouseProvider().sink(URI, {"table": "analytics.events"})
    message = str(error.value)
    assert "pip install 'daft[clickhouse]'" in message
    assert "write_sql" in message


def test_dtype_matrix_is_available_through_the_public_helper():
    matrix = dtype_matrix("clickhouse")
    assert matrix["Int64"] == "native"
    assert any(value == "reject" for value in matrix.values())
