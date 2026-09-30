from __future__ import annotations

import pytest

import daft
from daft import col
from daft.storage import open_uri, reset


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Keep the provider registry deterministic across tests."""
    reset()
    yield
    reset()


def test_daft_open_is_the_storage_entry_point():
    assert daft.open is open_uri
    assert daft.storage.open_uri is open_uri


def test_daft_open_reads_and_writes_end_to_end(tmp_path):
    path = str(tmp_path / "data.parquet")
    daft.open(path).write(daft.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    handle = daft.open(path)
    assert handle.read(filters=[col("a") > 1], columns=["b"]).to_pydict() == {"b": ["y", "z"]}
    assert handle.provider_info.name == "parquet"


def test_daft_open_reports_unknown_formats_actionably(tmp_path):
    from daft.storage import AmbiguousFormatError

    path = tmp_path / "data.bin"
    path.write_bytes(b"payload")
    with pytest.raises(AmbiguousFormatError, match="format="):
        daft.open(str(path))
