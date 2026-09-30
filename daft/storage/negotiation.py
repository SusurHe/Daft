from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from daft.storage.contracts import MetadataColumn
from daft.storage.residual import Residual

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Canonical pushdown negotiation order, matching Spark DataSource V2's documented order
#: (sample -> filter -> aggregate -> limit/top-n -> offset -> column pruning).
NEGOTIATION_ORDER: tuple[str, ...] = ("filters", "aggregation", "limit", "offset", "projection")


@runtime_checkable
class SupportsPushdownFilters(Protocol):
    """Source that can absorb predicates.

    Implementations must return every predicate they did not absorb as ``Residual.remaining`` so the
    engine can evaluate it above the scan. This is the contract already used internally by Daft's
    scan layer (``SupportsPushdownFilters::push_filters`` in ``src/daft-scan/src/pushdowns.rs``).
    """

    def push_filters(self, filters: Sequence[Any]) -> Residual[Any]:
        """Absorb what this source can evaluate itself and return the rest."""
        ...


@runtime_checkable
class SupportsPushdownProjection(Protocol):
    """Source that can prune columns before decoding."""

    def push_projection(self, columns: Sequence[str]) -> None:
        """Record the columns required by the query."""
        ...


@runtime_checkable
class SupportsPushdownLimit(Protocol):
    """Source that can stop reading after ``offset + limit`` rows."""

    def push_limit(self, limit: int, offset: int = 0) -> bool:
        """Absorb the limit and return whether it was accepted."""
        ...


@runtime_checkable
class SupportsPushdownAggregates(Protocol):
    """Source that can compute aggregates itself, for example a count pushdown."""

    def push_aggregation(self, aggregation: Any) -> Residual[Any]:
        """Absorb supported aggregates and return the rest."""
        ...


@runtime_checkable
class SupportsReportStatistics(Protocol):
    """Source that can report row counts and byte sizes before reading."""

    def report_statistics(self) -> Any | None:
        """Return table level statistics, or ``None`` when unavailable."""
        ...


@runtime_checkable
class SupportsReportPartitioning(Protocol):
    """Source that can report how its output is partitioned or sorted."""

    def report_partitioning(self) -> Any | None:
        """Return the output partitioning, or ``None`` when unavailable."""
        ...


@runtime_checkable
class SupportsMetadataColumns(Protocol):
    """Source that exposes metadata columns such as file path or row position."""

    def metadata_columns(self) -> Sequence[MetadataColumn]:
        """Return the metadata columns this source can produce."""
        ...


#: All pushdown related capability mixins, in negotiation order.
PUSHDOWN_MIXINS: tuple[tuple[str, type], ...] = (
    ("filters", SupportsPushdownFilters),
    ("aggregation", SupportsPushdownAggregates),
    ("limit", SupportsPushdownLimit),
    ("projection", SupportsPushdownProjection),
)

#: Reporting mixins that are not pushdowns but still part of the negotiated contract.
REPORTING_MIXINS: tuple[tuple[str, type], ...] = (
    ("statistics", SupportsReportStatistics),
    ("partitioning", SupportsReportPartitioning),
    ("metadata_columns", SupportsMetadataColumns),
)


@dataclass(frozen=True)
class ScanRequest:
    """What the engine would like the source to do.

    Attributes:
        filters: Predicates the engine would like to push down.
        columns: Columns required by the query, or ``None`` when everything is needed.
        limit: Row limit, or ``None``.
        offset: Rows to skip before applying the limit.
        aggregation: Aggregation expression the engine would like to push down.
    """

    filters: tuple[Any, ...] = ()
    columns: tuple[str, ...] | None = None
    limit: int | None = None
    offset: int = 0
    aggregation: Any | None = None

    @property
    def is_empty(self) -> bool:
        """Whether this request asks for nothing beyond a plain scan."""
        return not self.filters and self.columns is None and self.limit is None and self.aggregation is None


@dataclass
class ScanPlan:
    """Result of negotiating a :class:`ScanRequest` with a scan source.

    Attributes:
        filters: Predicates the source absorbed.
        filters_residual: Predicates the engine must evaluate itself.
        columns: Columns the source will produce, or ``None`` for all columns.
        limit: Limit the source absorbed, or ``None``.
        offset: Offset the source absorbed.
        aggregation: Aggregation the source absorbed, or ``None``.
        trace: Human readable log of the negotiation, surfaced by ``describe()``.
    """

    filters: tuple[Any, ...] = ()
    filters_residual: tuple[Any, ...] = ()
    columns: tuple[str, ...] | None = None
    limit: int | None = None
    offset: int = 0
    aggregation: Any | None = None
    trace: list[str] = field(default_factory=list)

    @property
    def fully_pushed_down(self) -> bool:
        """Whether the source absorbed every requested filter."""
        return not self.filters_residual

    def describe(self) -> str:
        """Render the plan as a multi-line string for diagnostics."""
        lines = list(self.trace) or ["no pushdown negotiation performed"]
        lines.append(f"filters: {len(self.filters)} pushed down, {len(self.filters_residual)} remaining")
        if self.columns is not None:
            lines.append(f"columns: {len(self.columns)} required")
        if self.limit is not None:
            lines.append(f"limit: {self.limit} (offset {self.offset})")
        if self.aggregation is not None:
            lines.append("aggregation: pushed down")
        return "\n".join(lines)


def negotiate(source: Any, request: ScanRequest) -> ScanPlan:
    """Run pushdown negotiation against a scan source.

    The source participates by implementing the ``Supports*`` mixins it cares about: an unimplemented
    mixin simply means "this source cannot absorb that operator", which is why capability detection is
    ``isinstance`` based rather than driven by a separate list of booleans that could drift from the
    implementation.

    Args:
        source: Scan source, typically produced by ``Provider.scan``.
        request: Operators the engine would like to push down.

    Returns:
        A :class:`ScanPlan` describing what was absorbed and what remains for the engine.
    """
    plan = ScanPlan(columns=request.columns)

    if request.filters:
        if isinstance(source, SupportsPushdownFilters):
            residual = source.push_filters(list(request.filters))
            plan.filters = tuple(residual.accepted)
            plan.filters_residual = tuple(residual.remaining)
            plan.trace.append(f"push_filters: {residual.describe()}")
        else:
            plan.filters_residual = tuple(request.filters)
            plan.trace.append("push_filters: unsupported, all filters remain")

    if request.aggregation is not None:
        if isinstance(source, SupportsPushdownAggregates):
            residual = source.push_aggregation(request.aggregation)
            accepted = residual.accepted[0] if residual.accepted else None
            plan.aggregation = accepted
            plan.trace.append(f"push_aggregation: {residual.describe()}")
        else:
            plan.trace.append("push_aggregation: unsupported, aggregation stays in the plan")

    if request.limit is not None:
        if isinstance(source, SupportsPushdownLimit) and source.push_limit(request.limit, request.offset):
            plan.limit = request.limit
            plan.offset = request.offset
            plan.trace.append(f"push_limit: accepted {request.limit} (offset {request.offset})")
        else:
            plan.trace.append("push_limit: unsupported, limit stays in the plan")

    if request.columns is not None:
        if isinstance(source, SupportsPushdownProjection):
            source.push_projection(list(request.columns))
            plan.trace.append(f"push_projection: {len(request.columns)} columns required")
        else:
            plan.columns = None
            plan.trace.append("push_projection: unsupported, all columns are read")

    return plan


def describe_source(source: Any) -> str:
    """Return a one-line summary of which capability mixins a scan source implements."""
    implemented = [name for name, mixin in (*PUSHDOWN_MIXINS, *REPORTING_MIXINS) if isinstance(source, mixin)]
    missing = [name for name, mixin in (*PUSHDOWN_MIXINS, *REPORTING_MIXINS) if not isinstance(source, mixin)]
    return f"supports: {implemented or 'none'} | unsupported: {missing or 'none'}"


__all__ = [
    "NEGOTIATION_ORDER",
    "PUSHDOWN_MIXINS",
    "REPORTING_MIXINS",
    "ScanPlan",
    "ScanRequest",
    "SupportsMetadataColumns",
    "SupportsPushdownAggregates",
    "SupportsPushdownFilters",
    "SupportsPushdownLimit",
    "SupportsPushdownProjection",
    "SupportsReportPartitioning",
    "SupportsReportStatistics",
    "describe_source",
    "negotiate",
]
