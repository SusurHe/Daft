from __future__ import annotations

import pytest

from daft.storage import Option, OptionError, OptionsContract


def contract() -> OptionsContract:
    """Build a contract exercising required, optional, choice and forwarded options."""
    return OptionsContract(
        required=(Option("host", doc="Server hostname"),),
        optional=(
            Option("port", type=int, default=8123, doc="Server port"),
            Option("mode", choices=("append", "overwrite"), default="append", doc="Write mode"),
        ),
        forwarded=("client.",),
    )


def test_defaults_and_required_values_are_applied():
    resolved = contract().validate({"host": "localhost"}, "test")
    assert resolved == {"host": "localhost", "port": 8123, "mode": "append"}


def test_missing_required_option_is_reported_with_doc():
    with pytest.raises(OptionError) as error:
        contract().validate({}, "test")
    message = str(error.value)
    assert "missing required option" in message
    assert "Server hostname" in message


def test_unknown_option_suggests_the_closest_name():
    with pytest.raises(OptionError) as error:
        contract().validate({"host": "localhost", "prot": 9000}, "test")
    message = str(error.value)
    assert "unknown option" in message
    assert "Did you mean 'port'" in message


def test_type_and_choice_validation():
    with pytest.raises(OptionError, match="expects int"):
        contract().validate({"host": "localhost", "port": "8123"}, "test")
    with pytest.raises(OptionError) as error:
        contract().validate({"host": "localhost", "mode": "appendd"}, "test")
    assert "must be one of" in str(error.value)
    assert "Did you mean 'append'?" in str(error.value)


def test_forwarded_options_bypass_validation():
    resolved = contract().validate({"host": "localhost", "client.timeout": 30}, "test")
    assert resolved["client.timeout"] == 30


def test_provider_contracts_reject_typos_end_to_end():
    from daft.storage import list_providers
    from daft.storage.registry import get

    parquet = get("parquet")
    assert "parquet" in [info.name for info in list_providers()]
    with pytest.raises(OptionError) as error:
        parquet.sink_options.validate({"compresion": "zstd"}, "parquet")
    assert "Did you mean 'compression'" in str(error.value)
