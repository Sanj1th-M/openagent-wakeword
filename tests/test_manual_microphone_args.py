import pytest

from tests.integration.manual_microphone_check import build_parser, threshold_arg


def test_threshold_arg_accepts_valid_values() -> None:
    assert threshold_arg("0.5") == 0.5
    assert threshold_arg("0.3") == 0.3


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "not-a-number"])
def test_threshold_arg_rejects_invalid_values(value: str) -> None:
    with pytest.raises(Exception):
        threshold_arg(value)


def test_parser_default_threshold_is_unchanged() -> None:
    parser = build_parser()

    args = parser.parse_args(["--models", ".models/openwakeword/v0.5.1"])

    assert args.threshold == 0.5
