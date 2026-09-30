from __future__ import annotations

import pytest

from daft.storage import (
    AmbiguousFormatError,
    ProviderKind,
    ProviderNotFoundError,
    describe_registry,
    infer_format,
    parse_uri,
    register,
    registered,
    registered_keys,
    reset,
    resolve_uri,
)


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Start every test from the built-in registry so ordering cannot leak state."""
    reset()
    yield
    reset()


def test_parse_uri_handles_schemes_local_paths_and_drive_letters():
    assert parse_uri("s3://bucket/key.parquet") == ("s3", "bucket/key.parquet")
    assert parse_uri("file:///tmp/data.parquet") == ("file", "/tmp/data.parquet")
    assert parse_uri("/tmp/data.parquet") == ("file", "/tmp/data.parquet")
    assert parse_uri("D:\\data\\file.parquet") == ("file", "D:\\data\\file.parquet")


def test_infer_format_uses_extensions():
    assert infer_format("s3://bucket/part-0.parquet") == "parquet"
    assert infer_format("/tmp/data.pq") == "parquet"
    assert infer_format("/tmp/events.jsonl") == "json"
    assert infer_format("/tmp/table.CSV") == "csv"
    assert infer_format("/tmp/unknown.bin") is None


def test_resolve_uri_splits_storage_and_format_axes():
    resolved = resolve_uri("s3://bucket/part-0.parquet")
    assert resolved.storage_key == "s3"
    assert resolved.storage.info.name == "s3"
    assert resolved.format_name == "parquet"
    assert resolved.format.info.name == "parquet"
    assert resolved.location.scheme == "s3"
    assert resolved.location.path == "bucket/part-0.parquet"
    assert "storage: 's3' -> s3" in resolved.trace[0]


def test_explicit_format_wins_over_extension():
    resolved = resolve_uri("s3://bucket/part-0.data", format="parquet")
    assert resolved.format_name == "parquet"
    assert "explicit" in resolved.trace[1]


def test_unknown_extension_raises_ambiguous_format_error():
    with pytest.raises(AmbiguousFormatError) as error:
        resolve_uri("s3://bucket/part-0.unknownext")
    assert "format=" in str(error.value)
    assert "parquet" in str(error.value)


def test_unknown_scheme_lists_registered_schemes():
    with pytest.raises(ProviderNotFoundError) as error:
        resolve_uri("ftp://host/data.parquet")
    message = str(error.value)
    assert "ftp" in message
    assert "s3" in message


def test_known_dependency_hint_is_attached():
    with pytest.raises(ProviderNotFoundError) as error:
        resolve_uri("clickhouse://host/db/table.parquet")
    assert "pip install 'daft[clickhouse]'" in str(error.value)


def test_duplicate_registration_requires_override():
    from daft.storage.providers import ParquetFormatProvider
    from daft.storage.registry import get

    register(get("parquet"))  # registering the identical object is an idempotent no-op
    with pytest.raises(ValueError, match="already registered"):
        register(ParquetFormatProvider())
    register(ParquetFormatProvider(), override=True)
    assert get("parquet").info.name == "parquet"


def test_registered_keys_and_describe_registry():
    assert "parquet" in registered_keys(ProviderKind.FILE_FORMAT)
    assert "file" in registered_keys(ProviderKind.FILESYSTEM)
    description = describe_registry()
    assert "file_format:" in description
    assert "parquet" in description


def test_registry_populates_itself_in_a_fresh_interpreter():
    """A brand new interpreter must resolve URIs without any explicit registration step."""
    import subprocess
    import sys

    code = (
        "from daft.storage import list_providers, open_uri;"
        "print(len(list_providers()));"
        "print(open_uri('/tmp/example.parquet').provider_info.name)"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    count, provider = completed.stdout.split()
    assert int(count) > 0
    assert provider == "parquet"


def test_reset_restores_builtins():
    reset(load_builtins=False)
    assert registered() == [], "an explicit reset must not be undone by lazy loading"
    reset()
    assert {info.name for info in registered()} >= {"parquet", "csv", "local", "legacy-datasource"}
