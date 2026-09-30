from __future__ import annotations

import pytest

from daft.storage import (
    DTypeSupport,
    NativeTabularSink,
    Option,
    OptionsContract,
    ProviderInfo,
    ProviderKind,
    TableCapability,
    TypeMapping,
    WriteProtocol,
    assert_conformance,
    check_provider,
    check_providers,
    list_providers,
    reset,
)
from daft.storage.conformance import _Kind
from daft.storage.negotiation import Residual
from daft.storage.registry import get


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Keep the provider registry deterministic across tests."""
    reset()
    yield
    reset()


class ConformingProvider:
    """Minimal provider that satisfies every conformance check."""

    info = ProviderInfo(
        name="conforming",
        kind=ProviderKind.FILE_FORMAT,
        keys=("conforming",),
        capabilities=frozenset({TableCapability.BATCH_READ, TableCapability.BATCH_WRITE}),
        write_protocol=WriteProtocol.ATOMIC_COMMIT,
        doc="Test double.",
    )
    type_mapping = TypeMapping(support=lambda dtype: DTypeSupport.NATIVE)
    scan_options = OptionsContract(optional=(Option("path", doc="Input path"),))
    sink_options = OptionsContract(optional=(Option("path", doc="Output path"),))

    def scan(self, uri, options):
        """Return a source that absorbs filters."""
        return _Source()

    def sink(self, uri, options):
        """Return a native sink specification."""
        return NativeTabularSink(root_dir=uri, file_format="conforming")


class _Source:
    applies_operators = False

    def read(self):  # pragma: no cover - never read in these tests
        """Not used."""
        raise NotImplementedError

    def push_filters(self, filters):
        """Absorb every predicate."""
        return Residual.accept_all(filters)

    def push_projection(self, columns):
        """Accept a projection."""
        self.columns = tuple(columns)


class DroppingSource(_Source):
    """Source that silently drops predicates, which is a correctness bug."""

    def push_filters(self, filters):
        """Drop everything."""
        return Residual(accepted=(), remaining=())


class UndocumentedOptionsProvider(ConformingProvider):
    """Provider whose option surface is incomplete."""

    info = ProviderInfo(
        name="undocumented",
        kind=ProviderKind.FILE_FORMAT,
        keys=("undocumented",),
        capabilities=frozenset({TableCapability.BATCH_READ}),
        doc="Test double.",
    )
    scan_options = OptionsContract(optional=(Option("path"),))
    sink_options = OptionsContract()


class NoProtocolProvider(ConformingProvider):
    """Provider that claims writes without declaring commit semantics."""

    info = ProviderInfo(
        name="no-protocol",
        kind=ProviderKind.FILE_FORMAT,
        keys=("no-protocol",),
        capabilities=frozenset({TableCapability.BATCH_WRITE}),
        doc="Test double.",
    )

    def sink(self, uri, options):  # pragma: no cover - reported before it is called
        """Never reached in these tests."""
        raise NotImplementedError


class SerializeWithoutStrategyProvider(ConformingProvider):
    """Provider that serialises some types but declares no fallback strategy."""

    info = ProviderInfo(
        name="serialize-without-strategy",
        kind=ProviderKind.FILE_FORMAT,
        keys=("serialize-without-strategy",),
        capabilities=frozenset({TableCapability.BATCH_READ}),
        doc="Test double.",
    )
    type_mapping = TypeMapping(
        support=lambda dtype: DTypeSupport.SERIALIZE if dtype.__class__ is _Kind else DTypeSupport.NATIVE,
        non_primitive="error",
    )
    sink_options = OptionsContract()


def test_conforming_provider_passes_every_check():
    assert check_provider(ConformingProvider(), sample_uri="file:///tmp/x.conforming") == []
    assert_conformance(ConformingProvider(), sample_uri="file:///tmp/x.conforming")


def test_duplicate_keys_are_reported():
    class DuplicateKeysProvider(ConformingProvider):
        info = ProviderInfo(
            name="duplicates",
            kind=ProviderKind.FILE_FORMAT,
            keys=("same", "same"),
            capabilities=frozenset({TableCapability.BATCH_READ}),
            doc="Test double.",
        )
        sink_options = OptionsContract()

    problems = check_provider(DuplicateKeysProvider())
    assert any("duplicate keys" in problem for problem in problems)


def test_missing_write_protocol_is_reported():
    problems = check_provider(NoProtocolProvider())
    assert any("no write_protocol" in problem for problem in problems)


def test_undocumented_options_are_reported():
    problems = check_provider(UndocumentedOptionsProvider())
    assert any("without a doc string" in problem for problem in problems)


def test_serialize_without_a_strategy_is_reported():
    problems = check_provider(SerializeWithoutStrategyProvider())
    assert any("non_primitive='error'" in problem for problem in problems)


def test_dropped_predicates_are_reported():
    class DroppingProvider(ConformingProvider):
        def scan(self, uri, options):
            """Return a source that drops predicates."""
            return DroppingSource()

    problems = check_provider(DroppingProvider(), sample_uri="file:///tmp/x.conforming")
    assert any("dropped a predicate" in problem for problem in problems)
    with pytest.raises(AssertionError, match="does not conform"):
        assert_conformance(DroppingProvider(), sample_uri="file:///tmp/x.conforming")


def test_scan_errors_are_reported_without_raising():
    class ExplodingProvider(ConformingProvider):
        def scan(self, uri, options):
            """Fail loudly to check that the kit reports instead of raising."""
            raise RuntimeError("boom")

    problems = check_provider(ExplodingProvider(), sample_uri="file:///tmp/x.conforming")
    assert any("RuntimeError: boom" in problem for problem in problems)


def test_builtin_providers_conform(tmp_path):
    parquet_file = tmp_path / "data.parquet"
    parquet_file.write_bytes(b"PAR1")
    samples = {
        "parquet": (str(parquet_file), {}),
        "csv": (str(tmp_path / "data.csv"), {}),
        "clickhouse": ("clickhouse://host:8123/db/table", {"url": "clickhouse://host:8123/db"}),
    }
    providers = [get(info.name) for info in list_providers()]
    results = check_providers([p for p in providers if p.info.name in samples], samples=samples)
    assert set(results) == {"parquet", "csv", "clickhouse"}
    for name, problems in results.items():
        assert problems == [], f"{name} does not conform: {problems}"
