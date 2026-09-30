from __future__ import annotations

from daft.storage import (
    Residual,
    ScanRequest,
    accounts_for_all,
    describe_source,
    negotiate,
)
from daft.storage.providers import CsvScanSource, ParquetScanSource


class PlainSource:
    """Scan source that implements no capability mixin at all."""

    applies_operators = False

    def read(self):  # pragma: no cover - never read in these tests
        """Not used: negotiation tests never read data."""
        raise NotImplementedError


def test_residual_helpers():
    assert Residual.accept_all([1, 2]).is_fully_pushed_down
    assert Residual.accept_none([1, 2]).remaining == (1, 2)
    assert Residual.partition([1, 2, 3, 4], lambda x: x % 2 == 0).accepted == (2, 4)
    merged = Residual(accepted=(1,), remaining=()).merge(Residual(accepted=(2,), remaining=(3,)))
    assert merged.accepted == (1, 2)
    assert merged.remaining == (3,)
    assert Residual(accepted=(1,), remaining=(2,)).describe() == "1 accepted / 1 remaining"


def test_accounts_for_all_detects_dropped_items():
    items = [object(), object()]
    assert accounts_for_all(items, Residual(accepted=(items[0],), remaining=(items[1],)))
    assert not accounts_for_all(items, Residual(accepted=(items[0],), remaining=()))
    assert not accounts_for_all(items, Residual(accepted=items, remaining=items))


def test_parquet_absorbs_filters_and_projection_but_not_limit():
    source = ParquetScanSource("s3://bucket/data.parquet", {})
    plan = negotiate(source, ScanRequest(filters=("a > 1",), columns=("a",), limit=10))
    assert plan.filters == ("a > 1",)
    assert plan.filters_residual == ()
    assert plan.limit is None
    assert source.required_columns == ("a",)
    assert "push_limit: unsupported" in "\n".join(plan.trace)
    assert plan.fully_pushed_down


def test_csv_declines_filters_and_limits_but_prunes_columns():
    source = CsvScanSource("s3://bucket/data.csv", {})
    plan = negotiate(source, ScanRequest(filters=("a > 1",), columns=("a",), limit=10))
    assert plan.filters == ()
    assert plan.filters_residual == ("a > 1",)
    assert source.required_columns == ("a",)
    assert not plan.fully_pushed_down
    assert "push_filters: unsupported" in "\n".join(plan.trace)


def test_source_without_mixins_keeps_everything_in_the_plan():
    plan = negotiate(PlainSource(), ScanRequest(filters=("a > 1",), columns=("a",), limit=5, offset=2))
    assert plan.filters_residual == ("a > 1",)
    assert plan.columns is None
    assert plan.limit is None
    assert len(plan.trace) == 3


def test_negotiation_never_loses_a_filter():
    filters = ("a > 1", "b < 2", "c is null")
    for source in (ParquetScanSource("x.parquet", {}), CsvScanSource("x.csv", {}), PlainSource()):
        plan = negotiate(source, ScanRequest(filters=filters))
        residual = Residual(accepted=plan.filters, remaining=plan.filters_residual)
        assert accounts_for_all(filters, residual), f"{type(source).__name__} dropped a filter"


def test_negotiation_order_follows_the_documented_sequence():
    from daft.storage.negotiation import NEGOTIATION_ORDER

    assert NEGOTIATION_ORDER == ("filters", "aggregation", "limit", "offset", "projection")
    plan = negotiate(ParquetScanSource("x.parquet", {}), ScanRequest(filters=("a",), limit=1, columns=("a",)))
    kinds = [line.split(":")[0] for line in plan.trace]
    assert kinds == ["push_filters", "push_limit", "push_projection"]


def test_describe_source_lists_supported_and_unsupported_mixins():
    description = describe_source(ParquetScanSource("x.parquet", {}))
    assert "filters" in description
    assert "limit" in description.split("|")[1]


def test_scan_plan_describe_is_human_readable():
    plan = negotiate(ParquetScanSource("x.parquet", {}), ScanRequest(filters=("a",), columns=("a",), limit=3))
    rendered = plan.describe()
    assert "filters: 1 pushed down, 0 remaining" in rendered
    assert "columns: 1 required" in rendered
