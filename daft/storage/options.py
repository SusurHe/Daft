from __future__ import annotations

import difflib
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from daft.storage.errors import OptionError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _closest(word: str, candidates: Sequence[str]) -> str | None:
    """Return the closest candidate to ``word``, used to build "did you mean" suggestions."""
    matches = difflib.get_close_matches(word, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


@dataclass(frozen=True)
class Option:
    """Declaration of a single provider option.

    Providers declare their options so that the engine can validate user input, apply defaults and
    produce actionable errors instead of silently ignoring typos. This follows the option contract
    used by Flink table factories (``requiredOptions`` / ``optionalOptions`` / ``forwardOptions``).

    Attributes:
        name: Option name as written by the user.
        type: Accepted Python type(s) for the value.
        required: Whether the option must be provided.
        choices: Optional allow-list of string values.
        default: Value used when the option is omitted.
        doc: One line description, surfaced in ``describe()`` output.
    """

    name: str
    type: type | tuple[type, ...] | None = None
    required: bool = False
    choices: tuple[str, ...] | None = None
    default: Any = None
    doc: str = ""

    def check(self, value: Any, provider: str) -> Any:
        """Validate a single value, returning it unchanged when it is acceptable."""
        if self.type is not None:
            expected = self.type if isinstance(self.type, tuple) else (self.type,)
            if value is not None and not isinstance(value, expected):
                names = " | ".join(t.__name__ for t in expected)
                raise OptionError(
                    provider,
                    f"{self.name!r} expects {names} but received {type(value).__name__}",
                    details=([self.doc] if self.doc else []),
                )
        if self.choices is not None and value is not None and value not in self.choices:
            closest = _closest(str(value), self.choices)
            raise OptionError(
                provider,
                f"{self.name!r} must be one of {list(self.choices)} but received {value!r}",
                details=([self.doc] if self.doc else []),
                suggestions=[f"Did you mean {closest!r}?"] if closest else [],
            )
        return value


@dataclass(frozen=True)
class OptionsContract:
    """The full option surface of one provider operation.

    Attributes:
        required: Options that must be supplied by the user.
        optional: Options that are validated when present and defaulted when absent.
        forwarded: Option names or prefixes (for example ``"client."``) that are passed through to
            the underlying client unchanged and therefore not validated. Flink calls this concept
            ``forwardOptions``.
    """

    required: tuple[Option, ...] = field(default_factory=tuple)
    optional: tuple[Option, ...] = field(default_factory=tuple)
    forwarded: tuple[str, ...] = field(default_factory=tuple)

    @property
    def declared(self) -> tuple[Option, ...]:
        """All explicitly declared options.

        Options listed under ``required`` are marked required here so that the contract itself is the
        single source of truth; an option declared in ``optional`` may still set ``required=True``.
        """
        return tuple(replace(option, required=True) for option in self.required) + self.optional

    @property
    def names(self) -> tuple[str, ...]:
        """Names of all explicitly declared options, sorted."""
        return tuple(sorted(option.name for option in self.declared))

    def is_forwarded(self, name: str) -> bool:
        """Whether an option name matches a forwarded name or prefix."""
        return any(name == prefix or name.startswith(prefix) for prefix in self.forwarded)

    def validate(self, options: Mapping[str, Any], provider: str) -> dict[str, Any]:
        """Validate user supplied options and apply defaults.

        Args:
            options: Options as passed by the user.
            provider: Provider name, used in error messages.

        Returns:
            A new mapping containing the validated options plus defaults.

        Raises:
            OptionError: If required options are missing, unknown options are present, or a value
                has the wrong type or an unsupported choice.
        """
        declared = {option.name: option for option in self.declared}

        missing = [name for name, option in declared.items() if option.required and name not in options]
        if missing:
            raise OptionError(
                provider,
                f"missing required option(s) {sorted(missing)}",
                details=[f"{name}: {declared[name].doc}" for name in sorted(missing) if declared[name].doc],
                suggestions=[f"Accepted options: {list(self.names)}"],
            )

        unknown = [name for name in options if name not in declared and not self.is_forwarded(name)]
        if unknown:
            suggestions = []
            for name in sorted(unknown):
                closest = _closest(name, self.names)
                suggestions.append(
                    f"Did you mean {closest!r} instead of {name!r}?" if closest else f"Unknown option {name!r}"
                )
            raise OptionError(
                provider,
                f"unknown option(s) {sorted(unknown)}",
                details=[f"Accepted options: {list(self.names)}"]
                + ([f"Forwarded options: {list(self.forwarded)}"] if self.forwarded else []),
                suggestions=suggestions,
            )

        resolved: dict[str, Any] = {}
        for name, option in declared.items():
            if name in options:
                resolved[name] = option.check(options[name], provider)
            elif option.default is not None:
                resolved[name] = option.default
            elif option.required:
                raise OptionError(provider, f"missing required option {name!r}")
        for name, value in options.items():
            if self.is_forwarded(name):
                resolved[name] = value
        return resolved
