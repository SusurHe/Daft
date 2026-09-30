from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from daft.storage.contracts import (
    DTypeSupport,
    SupportsScan,
    SupportsSink,
    TypeMapping,
)
from daft.storage.options import OptionsContract
from daft.storage.residual import Residual, accounts_for_all

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class _Kind:
    """Stand-in data type used to probe a provider's type mapping without the engine runtime."""

    def __init__(self, kind: str) -> None:
        self._kind = kind

    def __getattr__(self, name: str) -> Any:
        if name.startswith("is_"):
            return lambda: name[3:] == self._kind
        raise AttributeError(name)

    def __str__(self) -> str:
        return f"Fake[{self._kind}]"


#: One probe per data type kind that ``dtype_kind`` can return.
_PROBE_KINDS: tuple[str, ...] = (
    "primitive",
    "python",
    "list",
    "struct",
    "map",
    "tensor",
    "sparse_tensor",
    "image",
    "embedding",
    "file",
)


class _Predicate:
    """Opaque predicate used to check that pushdown negotiation never drops an operator."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


def check_provider(
    provider: Any,
    *,
    sample_uri: str | None = None,
    sample_options: Mapping[str, Any] | None = None,
) -> list[str]:
    """Check that a provider honours the storage provider contract.

    The checks cover the properties the design relies on and that are easy to get wrong: declared
    capabilities must match the implemented methods, option surfaces must be complete and documented,
    pushdown negotiation must never drop an operator, and the type mapping must be self consistent.

    Args:
        provider: Provider object to check.
        sample_uri: Optional URI used to exercise ``scan()``. Checks that need to build a scan source
            are skipped when it is omitted.
        sample_options: Options passed to ``scan()`` together with ``sample_uri``.

    Returns:
        A list of human readable problems. An empty list means the provider conforms.
    """
    problems: list[str] = []
    problems += _check_identity(provider)
    problems += _check_options(provider, "scan_options")
    problems += _check_options(provider, "sink_options")
    problems += _check_write_declaration(provider)
    problems += _check_type_mapping(provider)
    problems += _check_read_path(provider, sample_uri, sample_options or {})
    return problems


def assert_conformance(
    provider: Any,
    *,
    sample_uri: str | None = None,
    sample_options: Mapping[str, Any] | None = None,
) -> None:
    """Raise ``AssertionError`` listing every conformance problem of a provider."""
    problems = check_provider(provider, sample_uri=sample_uri, sample_options=sample_options)
    if problems:
        name = getattr(getattr(provider, "info", None), "name", type(provider).__name__)
        details = "\n".join(f"  - {problem}" for problem in problems)
        raise AssertionError(f"provider {name!r} does not conform to the storage provider contract:\n{details}")


def check_providers(
    providers: Iterable[Any],
    *,
    samples: Mapping[str, tuple[str, Mapping[str, Any]]] | None = None,
) -> dict[str, list[str]]:
    """Check several providers at once.

    Args:
        providers: Providers to check.
        samples: Optional mapping from provider name to ``(uri, options)`` used for scan checks.

    Returns:
        A mapping from provider name to its problems. Providers without problems are included with an
        empty list so callers can report coverage.
    """
    samples = samples or {}
    results: dict[str, list[str]] = {}
    for provider in providers:
        name = provider.info.name
        uri, options = samples.get(name, (None, {}))
        results[name] = check_provider(provider, sample_uri=uri, sample_options=options)
    return results


def _check_identity(provider: Any) -> list[str]:
    info = getattr(provider, "info", None)
    if info is None:
        return [f"{type(provider).__name__}: missing 'info'"]
    problems: list[str] = []
    if not info.name:
        problems.append("info.name must not be empty")
    if not info.keys:
        problems.append(f"{info.name}: info.keys must not be empty")
    duplicated = [key for key, count in Counter(info.keys).items() if count > 1]
    if duplicated:
        problems.append(f"{info.name}: duplicate keys {duplicated}")
    lowered = [key for key in info.keys if key != key.lower()]
    if lowered:
        problems.append(f"{info.name}: keys must be lower case, found {lowered}")
    return problems


def _check_options(provider: Any, attribute: str) -> list[str]:
    name = getattr(getattr(provider, "info", None), "name", type(provider).__name__)
    contract = getattr(provider, attribute, None)
    if not isinstance(contract, OptionsContract):
        return [f"{name}: {attribute} must be an OptionsContract"]
    problems: list[str] = []
    duplicated = [option for option, count in Counter(o.name for o in contract.declared).items() if count > 1]
    if duplicated:
        problems.append(f"{name}: {attribute} declares options twice: {duplicated}")
    undocumented = [option.name for option in contract.declared if not option.doc]
    if undocumented:
        problems.append(f"{name}: {attribute} options without a doc string: {undocumented}")
    for prefix in contract.forwarded:
        if prefix.endswith(".") and len(prefix) == 1:
            problems.append(f"{name}: {attribute} forwards every option with {prefix!r}; list the actual prefix")
    return problems


def _check_write_declaration(provider: Any) -> list[str]:
    info = getattr(provider, "info", None)
    if info is None:
        return []
    problems: list[str] = []
    if info.can_write():
        if info.write_protocol is None:
            problems.append(f"{info.name}: declares BATCH_WRITE but no write_protocol")
        if not isinstance(provider, SupportsSink):
            problems.append(f"{info.name}: declares BATCH_WRITE but does not implement sink()")
    elif info.write_protocol is not None:
        problems.append(f"{info.name}: declares a write_protocol but not BATCH_WRITE")
    if info.can_read() and not isinstance(provider, SupportsScan) and info.api_level.value == "v2":
        problems.append(f"{info.name}: declares BATCH_READ but does not implement scan()")
    return problems


def _check_type_mapping(provider: Any) -> list[str]:
    name = getattr(getattr(provider, "info", None), "name", type(provider).__name__)
    mapping = getattr(provider, "type_mapping", None)
    if not isinstance(mapping, TypeMapping):
        return [f"{name}: type_mapping must be a TypeMapping"]
    problems: list[str] = []
    if mapping.non_primitive not in ("error", "str", "bytes"):
        problems.append(f"{name}: non_primitive must be 'error', 'str' or 'bytes', found {mapping.non_primitive!r}")
    supports = {mapping.check(_Kind(kind)) for kind in _PROBE_KINDS}
    invalid = [value for value in supports if not isinstance(value, DTypeSupport)]
    if invalid:
        problems.append(f"{name}: type_mapping.support must return a DTypeSupport, found {invalid}")
    if DTypeSupport.SERIALIZE in supports and mapping.non_primitive == "error":
        problems.append(f"{name}: maps some types to SERIALIZE but declares non_primitive='error'")
    return problems


def _check_read_path(provider: Any, sample_uri: str | None, sample_options: Mapping[str, Any]) -> list[str]:
    info = getattr(provider, "info", None)
    if info is None or not info.can_read() or not isinstance(provider, SupportsScan) or sample_uri is None:
        return []
    problems: list[str] = []
    try:
        source = provider.scan(sample_uri, dict(sample_options))
    except Exception as error:  # noqa: BLE001 - reported as a conformance problem
        return [f"{info.name}: scan({sample_uri!r}) raised {type(error).__name__}: {error}"]
    if not hasattr(source, "read"):
        problems.append(f"{info.name}: scan() must return a source exposing read()")

    predicates: tuple[_Predicate, ...] = (_Predicate("a"), _Predicate("b"))
    request_columns: Sequence[str] = ("a",)
    from daft.storage.negotiation import ScanRequest, negotiate

    plan = negotiate(source, ScanRequest(filters=predicates, columns=request_columns))
    residual = Residual(accepted=plan.filters, remaining=plan.filters_residual)
    if not accounts_for_all(predicates, residual):
        problems.append(f"{info.name}: pushdown negotiation dropped a predicate, which breaks correctness")
    if plan.columns is None and hasattr(source, "push_projection"):
        problems.append(f"{info.name}: implements push_projection but negotiation did not record columns")
    return problems


__all__ = ["assert_conformance", "check_provider", "check_providers"]
