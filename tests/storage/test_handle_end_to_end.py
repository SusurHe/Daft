from __future__ import annotations

import pytest

import daft
from daft import col
from daft.storage import (
    AmbiguousFormatError,
    ScanRequest,
    describe_uri,
    dtype_matrix,
    list_providers,
    open_uri,
    reset,
    sink_form,
)


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Keep the provider registry deterministic across tests."""
    reset()
    yield
    reset()


@pytest.fixture
def parquet_path(tmp_path):
    """Write a small parquet dataset and return its path."""
    path = tmp_path / "data.parquet"
    daft.from_pydict({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]}).write_parquet(str(path))
    return str(path)


@pytest.fixture
def csv_path(tmp_path):
    """Write a small csv dataset and return its path."""
    path = tmp_path / "data.csv"
    daft.from_pydict({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]}).write_csv(str(path))
    return str(path)


def test_handle_read_matches_read_parquet(parquet_path):
    handle = open_uri(parquet_path)
    assert handle.format_name == "parquet"
    assert handle.provider_info.name == "parquet"
    assert handle.read().to_pydict() == daft.read_parquet(parquet_path).to_pydict()


def test_operators_are_applied_even_when_not_pushed_down(parquet_path):
    handle = open_uri(parquet_path)
    assert handle.read(filters=[col("a") > 1], columns=["b"], limit=2).to_pydict() == {"b": ["x", "y"]}
    assert handle.read(columns=["b"], limit=1, offset=3).to_pydict() == {"b": ["z"]}


def test_plan_reports_pushdown_decisions(parquet_path):
    handle = open_uri(parquet_path)
    plan = handle.plan(ScanRequest(filters=(col("a") > 1,), columns=("a",), limit=5))
    assert len(plan.filters) == 1
    assert not plan.filters_residual
    assert plan.limit is None  # parquet does not absorb limits, matching can_absorb_limit == false


def test_csv_capabilities_are_honest(csv_path):
    handle = open_uri(csv_path)
    plan = handle.plan(ScanRequest(filters=(col("a") > 1,)))
    assert plan.filters_residual, "csv cannot absorb predicates, so the filter must remain"
    assert handle.read(filters=[col("a") > 1]).to_pydict() == {"a": [2, 3, 4], "b": ["x", "y", "z"]}


def test_sink_spec_uses_the_native_writer(tmp_path):
    handle = open_uri(str(tmp_path / "out.parquet"))
    spec = handle.sink(compression="zstd", partition_cols=("b",))
    assert sink_form(spec) == "native_tabular"
    assert spec.file_format == "parquet"
    assert spec.format_options == {"compression": "zstd"}


def test_native_sink_writes_through_the_handle(tmp_path):
    out = str(tmp_path / "out.parquet")
    open_uri(out).write(daft.from_pydict({"a": [1, 2]}))
    assert daft.read_parquet(out).to_pydict() == {"a": [1, 2]}

    csv_out = str(tmp_path / "out.csv")
    open_uri(csv_out).write(daft.from_pydict({"a": [3, 4]}))
    assert daft.read_csv(csv_out).to_pydict() == {"a": [3, 4]}


def test_v1_fallback_backend_skips_negotiation_and_records_a_hint():
    class StubSource:
        """Legacy source that only implements the DataSource contract."""

        applies_operators = False

        def read(self):
            """Return a tiny DataFrame."""
            return daft.from_pydict({"a": [1, 2, 3]})

    handle = open_uri("/tmp/stub", format="datasource", source=StubSource())
    plan = handle.plan(ScanRequest(filters=(col("a") > 1,)))
    assert any("v1 fallback" in line for line in plan.trace)
    assert handle.read(filters=[col("a") > 1]).to_pydict() == {"a": [2, 3]}


def test_describe_explains_layers_location_and_protocol(parquet_path):
    text = describe_uri(parquet_path)
    assert "source=user" in text
    assert "layers: ['filesystem', 'file_format']" in text
    assert "write_protocol: atomic_commit" in text
    assert "capabilities: ['batch_read', 'batch_write']" in text


def test_unknown_extension_requires_explicit_format(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"not really data")
    with pytest.raises(AmbiguousFormatError):
        open_uri(str(path))
    assert open_uri(str(path), format="csv").format_name == "csv"


def test_dtype_matrix_uses_real_daft_types():
    matrix = dtype_matrix("parquet")
    assert matrix["Int64"] == "native"
    assert all(value in {"native", "serialize", "reject"} for value in matrix.values())
    csv_matrix = dtype_matrix("csv")
    assert csv_matrix["Int64"] == "native"
    assert any(value == "reject" for value in csv_matrix.values())


def test_registered_providers_are_listable():
    names = {info.name for info in list_providers()}
    assert {"local", "s3", "parquet", "csv", "legacy-datasource"} <= names
