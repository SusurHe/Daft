from __future__ import annotations

import contextlib
import io

import pytest

import daft
from daft.storage.legacy import first_path, tabular_scan_configs


@pytest.fixture
def parquet_path(tmp_path):
    """Write a small parquet dataset and return its path."""
    path = str(tmp_path / "data.parquet")
    daft.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]}).write_parquet(path)
    return path


def test_first_path_accepts_a_single_path_or_a_sequence():
    assert first_path("/tmp/a.parquet") == "/tmp/a.parquet"
    assert first_path(["/tmp/a.parquet", "/tmp/b.parquet"]) == "/tmp/a.parquet"
    with pytest.raises(ValueError, match="at least one path"):
        first_path([])


def test_tabular_scan_configs_builds_configs_for_a_parquet_path(parquet_path):
    file_format_config, storage_config = tabular_scan_configs(
        parquet_path,
        format="parquet",
        multithreaded_io=True,
        io_config=None,
        coerce_int96_timestamp_unit=None,
        row_groups=None,
        chunk_size=None,
        ignore_corrupt_files=False,
    )
    assert file_format_config is not None
    assert storage_config is not None


def test_read_parquet_still_reads_and_filters(parquet_path):
    df = daft.read_parquet(parquet_path).where(daft.col("a") > 1)
    assert df.to_pydict() == {"a": [2, 3], "b": ["y", "z"]}


def test_read_parquet_keeps_the_scan_operator_and_pushdown(parquet_path):
    """The rewritten entry point must still produce a glob scan with the predicate pushed into it."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        daft.read_parquet(parquet_path).where(daft.col("a") > 1).explain(show_all=True)
    plan = buffer.getvalue()
    assert "GlobScan" in plan
    assert "Filter pushdown" in plan
    assert "Pushdowns: {filter:" in plan
