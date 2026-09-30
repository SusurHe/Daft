from __future__ import annotations

import pytest

from daft.storage import (
    DTypeSupport,
    NativeTabularSink,
    TableCapability,
    UnsupportedOperationError,
    WriteProtocol,
    dtype_matrix,
    precheck_dtypes,
    sink_form,
)
from daft.storage.negotiation import SupportsPushdownFilters, SupportsPushdownProjection
from daft.storage.providers import (
    CsvFormatProvider,
    CsvScanSource,
    LegacyDataSourceProvider,
    ParquetFormatProvider,
    ParquetScanSource,
    dtype_kind,
)


class FakeTensor:
    """Minimal stand-in for a multimodal Daft data type."""

    def is_tensor(self) -> bool:
        """Report as a tensor type."""
        return True

    def __str__(self) -> str:
        """Render a readable name for matrix output."""
        return "Tensor"


class FakePrimitive:
    """Minimal stand-in for a primitive Daft data type."""

    def __str__(self) -> str:
        """Render a readable name for matrix output."""
        return "Int64"


def test_dtype_kind_uses_duck_typing():
    assert dtype_kind(FakeTensor()) == "tensor"
    assert dtype_kind(FakePrimitive()) == "primitive"


def test_parquet_supports_multimodal_types_csv_does_not():
    parquet, csv = ParquetFormatProvider(), CsvFormatProvider()
    assert parquet.type_mapping.check(FakeTensor()) is DTypeSupport.NATIVE
    assert parquet.type_mapping.check(FakePrimitive()) is DTypeSupport.NATIVE
    assert csv.type_mapping.check(FakePrimitive()) is DTypeSupport.NATIVE
    assert csv.type_mapping.check(FakeTensor()) is DTypeSupport.REJECT


def test_precheck_dtypes_fails_fast_with_alternatives():
    csv = CsvFormatProvider()
    assert csv.type_mapping.rejected([("a", FakePrimitive())]) == []
    with pytest.raises(UnsupportedOperationError) as error:
        precheck_dtypes("csv", [("a", FakePrimitive()), ("img", FakeTensor())], csv.type_mapping)
    message = str(error.value)
    assert "img" in message
    assert "Alternatives:" in message
    assert error.value.alternatives


def test_type_mapping_matrix_is_renderable():
    matrix = ParquetFormatProvider().type_mapping.matrix([FakePrimitive(), FakeTensor()])
    assert matrix == {"Int64": "native", "Tensor": "native"}


def test_provider_capabilities_and_write_protocols():
    parquet, csv = ParquetFormatProvider(), CsvFormatProvider()
    for provider in (parquet, csv):
        assert provider.info.can_read()
        assert provider.info.can_write()
        assert provider.info.write_protocol is WriteProtocol.ATOMIC_COMMIT
        assert TableCapability.BATCH_READ in provider.info.capabilities
    assert ParquetFormatProvider().info.keys == ("parquet", "pq")


def test_write_protocol_semantics():
    assert WriteProtocol.ATOMIC_COMMIT.is_atomic
    assert WriteProtocol.ATOMIC_COMMIT.supports_abort
    assert WriteProtocol.APPEND_ONLY.may_duplicate_on_retry
    assert not WriteProtocol.APPEND_ONLY.supports_abort


def test_sink_forms_map_to_existing_engine_paths():
    sink = ParquetFormatProvider().sink("s3://bucket/out", {"compression": "zstd", "partition_cols": ("d",)})
    assert sink_form(sink) == "native_tabular"
    assert isinstance(sink, NativeTabularSink)
    assert sink.format_options == {"compression": "zstd"}
    assert sink.partition_cols == ("d",)


def test_scan_sources_only_advertise_what_they_do():
    parquet, csv = ParquetScanSource("x.parquet", {}), CsvScanSource("x.csv", {})
    assert isinstance(parquet, SupportsPushdownFilters)
    assert isinstance(parquet, SupportsPushdownProjection)
    assert not isinstance(csv, SupportsPushdownFilters)
    assert isinstance(csv, SupportsPushdownProjection)


def test_legacy_provider_declares_v1_fallback_and_requires_source_option():
    provider = LegacyDataSourceProvider()
    assert provider.info.api_level.value == "v1_fallback"
    with pytest.raises(Exception, match="missing required option"):
        provider.scan_options.validate({}, provider.info.name)
    with pytest.raises(UnsupportedOperationError):
        provider.sink("datasource://x", {})


def test_dtype_matrix_helper_uses_the_provider_mapping():
    matrix = dtype_matrix("parquet", [FakePrimitive(), FakeTensor()])
    assert matrix["Tensor"] == "native"
