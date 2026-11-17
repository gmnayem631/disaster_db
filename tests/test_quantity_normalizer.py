"""Regression tests for quantity_normalizer.normalize_quantity."""
import pytest

from extraction.branch_b.quantity_normalizer import normalize_quantity


@pytest.mark.parametrize("raw, expected", [
    ("two lakh", 200_000),
    ("three crore", 30_000_000),
    ("twenty five", 25),
    ("one hundred and twenty five", 125),
    ("58 lakh", 5_800_000),
    ("5, 40,510", 540_510),
])
def test_point_values(raw, expected):
    assert normalize_quantity(raw).normalized_value == expected


def test_range_keeps_both_bounds():
    q = normalize_quantity("10 to 15 lakh")
    assert q.normalized_value is None
    assert q.lower_bound == 1_000_000
    assert q.upper_bound == 1_500_000
