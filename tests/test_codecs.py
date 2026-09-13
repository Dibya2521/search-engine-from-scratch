"""Tests for variable-byte and delta encoding."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.codecs import (
    CodecError,
    decode,
    decode_sorted,
    decode_sorted_at,
    encode,
    encode_number,
    encode_sorted,
    from_deltas,
    to_deltas,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, b"\x80"),
        (1, b"\x81"),
        (127, b"\xff"),
        # 128 is the first value needing two bytes: 1 in the high chunk, 0 in
        # the low one, with the terminator set on the last byte.
        (128, b"\x01\x80"),
        (129, b"\x01\x81"),
        (16_383, b"\x7f\xff"),
        (16_384, b"\x01\x00\x80"),
    ],
)
def test_encoding_one_number(value: int, expected: bytes) -> None:
    assert encode_number(value) == expected


@pytest.mark.parametrize(
    ("value", "width"),
    [(0, 1), (127, 1), (128, 2), (16_383, 2), (16_384, 3), (2_097_151, 3)],
)
def test_a_small_number_takes_fewer_bytes(value: int, width: int) -> None:
    """A smaller value takes fewer bytes, which is what varbyte buys."""
    assert len(encode_number(value)) == width


def test_a_negative_number_cannot_be_encoded() -> None:
    with pytest.raises(CodecError, match="cannot represent"):
        encode_number(-1)


def test_encoding_nothing_produces_nothing() -> None:
    assert encode([]) == b""
    assert decode(b"") == []


def test_a_sequence_round_trips() -> None:
    values = [0, 1, 127, 128, 300, 16_384, 1_000_000]
    assert decode(encode(values)) == values


def test_truncated_input_is_rejected() -> None:
    """Without the check this decodes to a plausible smaller number."""
    truncated = encode_number(16_384)[:-1]
    with pytest.raises(CodecError, match="ends in the middle"):
        decode(truncated)


def test_deltas_are_the_gaps() -> None:
    assert to_deltas([3, 8, 12, 40]) == [3, 5, 4, 28]


def test_deltas_of_nothing() -> None:
    assert to_deltas([]) == []
    assert from_deltas([]) == []


def test_repeated_values_give_zero_gaps() -> None:
    """Sorted allows equal neighbours, and a zero gap is one byte."""
    assert to_deltas([5, 5, 5]) == [5, 0, 0]


def test_unsorted_values_are_rejected() -> None:
    with pytest.raises(CodecError, match="sorted ascending"):
        to_deltas([5, 3])


def test_deltas_round_trip() -> None:
    values = [0, 2, 5, 9, 400]
    assert from_deltas(to_deltas(values)) == values


def test_the_combination_round_trips() -> None:
    values = [0, 2, 5, 9, 400, 100_000]
    assert decode_sorted(encode_sorted(values)) == values


def test_sorted_encoding_beats_plain_encoding_on_large_values() -> None:
    """Why delta encoding comes first: it is what makes varbyte pay."""
    values = list(range(1_000_000, 1_000_100))
    assert len(encode_sorted(values)) < len(encode(values))


def test_delta_encoding_does_not_help_when_gaps_are_large() -> None:
    """The falsifying case, and the reason to measure rather than assume.

    Values spaced further apart than they are large make the gaps no smaller
    than the values, so the technique buys nothing. Compression is a property
    of the data, not of the code.
    """
    values = [0, 200_000, 400_000, 600_000]
    assert len(encode_sorted(values)) >= len(encode(values))


@given(st.lists(st.integers(0, 2**40), max_size=60))
def test_any_sequence_round_trips(values: list[int]) -> None:
    assert decode(encode(values)) == values


@given(st.lists(st.integers(0, 2**32), max_size=60))
def test_any_sorted_sequence_round_trips(values: list[int]) -> None:
    ordered = sorted(values)
    assert decode_sorted(encode_sorted(ordered)) == ordered


@given(st.lists(st.integers(0, 2**20), min_size=1, max_size=60))
def test_encoded_bytes_are_never_longer_than_eight_per_value(
    values: list[int],
) -> None:
    """A cap that holds however large the values are, up to the tested range."""
    assert len(encode(values)) <= 8 * len(values)


@given(st.integers(0, 2**60))
def test_every_byte_but_the_last_is_unterminated(value: int) -> None:
    """The invariant a decoder relies on to know where a number ends."""
    encoded = encode_number(value)
    assert all(byte < 0x80 for byte in encoded[:-1])
    assert encoded[-1] >= 0x80


def test_bulk_decoding_reads_exactly_the_values_asked_for() -> None:
    encoded = encode_sorted([0, 5, 9]) + encode_sorted([400])
    values, after = decode_sorted_at(encoded, 0, 3)
    assert values == [0, 5, 9]
    assert after == len(encode_sorted([0, 5, 9]))


def test_bulk_decoding_handles_gaps_wider_than_one_byte() -> None:
    """The multi-byte branch: a gap of 100,000 needs three bytes."""
    values = [0, 5, 100_005, 100_006]
    encoded = encode_sorted(values)
    assert decode_sorted_at(encoded, 0, len(values)) == (values, len(encoded))


def test_bulk_decoding_rejects_a_buffer_that_runs_out() -> None:
    with pytest.raises(CodecError, match="middle of a number"):
        decode_sorted_at(encode_sorted([1, 2]), 0, 5)


def test_bulk_decoding_nothing_reads_nothing() -> None:
    assert decode_sorted_at(b"", 0, 0) == ([], 0)


def test_encoding_sorted_rejects_unsorted_values() -> None:
    """The fast path checks the order itself rather than delegating."""
    with pytest.raises(CodecError, match="sorted ascending"):
        encode_sorted([5, 3])
