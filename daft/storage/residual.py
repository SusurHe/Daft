from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

T = TypeVar("T")


@dataclass(frozen=True)
class Residual(Generic[T]):
    """Outcome of a pushdown negotiation round.

    Providers accept the items they can handle and must return everything else as ``remaining`` so
    that the engine can re-evaluate those items above the scan. This mirrors the contract used by
    Spark DataSource V2 (``SupportsPushDownFilters`` returns the filters it could *not* handle) and
    Flink connectors (``FilterPushdownResult`` exposes accepted and remaining filters). It is also
    already the internal convention in Daft for filter pushdown:
    ``src/daft-scan/src/pushdowns.rs`` defines
    ``fn push_filters(&self, filter: &[ExprRef]) -> (Vec<ExprRef>, Vec<ExprRef>)``.

    Attributes:
        accepted: Items the provider has taken responsibility for.
        remaining: Items the engine still has to evaluate itself.
    """

    accepted: tuple[T, ...] = field(default_factory=tuple)
    remaining: tuple[T, ...] = field(default_factory=tuple)

    @classmethod
    def accept_all(cls, items: Iterable[T]) -> Residual[T]:
        """Accept every item, leaving nothing for the engine to evaluate."""
        return cls(accepted=tuple(items), remaining=())

    @classmethod
    def accept_none(cls, items: Iterable[T]) -> Residual[T]:
        """Reject every item, leaving all of them for the engine to evaluate."""
        return cls(accepted=(), remaining=tuple(items))

    @classmethod
    def partition(cls, items: Iterable[T], predicate) -> Residual[T]:
        """Split items into accepted (``predicate`` is true) and remaining (predicate is false)."""
        accepted: list[T] = []
        remaining: list[T] = []
        for item in items:
            (accepted if predicate(item) else remaining).append(item)
        return cls(accepted=tuple(accepted), remaining=tuple(remaining))

    @property
    def is_fully_pushed_down(self) -> bool:
        """Whether the provider accepted every item in this round."""
        return not self.remaining

    def __bool__(self) -> bool:
        """Return whether the provider accepted at least one item."""
        return bool(self.accepted)

    def __iter__(self) -> Iterator[T]:
        """Iterate over the accepted items."""
        return iter(self.accepted)

    def __len__(self) -> int:
        """Total number of items seen in this negotiation round."""
        return len(self.accepted) + len(self.remaining)

    def merge(self, other: Residual[T]) -> Residual[T]:
        """Combine two rounds of negotiation for the same pushdown kind."""
        return Residual(accepted=self.accepted + other.accepted, remaining=self.remaining + other.remaining)

    def describe(self) -> str:
        """Render a short summary such as ``2 accepted / 1 remaining``."""
        return f"{len(self.accepted)} accepted / {len(self.remaining)} remaining"


def accounts_for_all(requested: Sequence[T], residual: Residual[T]) -> bool:
    """Return whether a negotiation round accounted for every requested item exactly once.

    This is the no-loss property that the conformance kit asserts for every provider: a pushdown
    round may decline items, but it must never drop them.
    """
    seen: dict[int, int] = {}
    for item in (*residual.accepted, *residual.remaining):
        seen[id(item)] = seen.get(id(item), 0) + 1
    if len(seen) != len({id(item) for item in requested}):
        return False
    return all(seen.get(id(item), 0) == 1 for item in requested)
