"""Regression tests for candidate_generator: a number keeps its modifier
("at least", "nearly", "more than"...) in every number format."""
import pytest

from extraction.branch_b.candidate_generator import generate_candidates
from extraction.branch_b.quantity_normalizer import normalize_quantity


def quantities(text):
    _, candidates = generate_candidates(text)
    return [c for c in candidates if c.kind == "QUANTITY"]


@pytest.mark.parametrize("text, raw, value, modifier", [
    # lakh / crore / million: the modifier used to be dropped
    ("At least 58 lakh people have been affected.", "At least 58 lakh", 5_800_000, "AT_LEAST"),
    ("Nearly 4.5 crore taka was allocated.", "Nearly 4.5 crore", 45_000_000, "NEARLY"),
    ("More than 2 million people were marooned.", "More than 2 million", 2_000_000, "MORE_THAN"),
    # word numbers: the modifier used to be dropped
    ("At least three people died.", "At least three", 3, "AT_LEAST"),
    ("More than two lakh people were displaced.", "More than two lakh", 200_000, "MORE_THAN"),
    ("Over one hundred and twenty five houses collapsed.",
     "Over one hundred and twenty five", 125, "OVER"),
    # these already worked and must keep working
    ("At least 31 people died.", "At least 31", 31, "AT_LEAST"),
    ("Around 10,000 families remained marooned.", "Around 10,000", 10_000, "AROUND"),
    ("Nearly 12,27,554 people were affected.", "Nearly 12,27,554", 1_227_554, "NEARLY"),
    ("About 5, 40,510 people were affected.", "About 5, 40,510", 540_510, "ABOUT"),
])
def test_number_keeps_its_modifier(text, raw, value, modifier):
    [candidate] = quantities(text)            # exactly one candidate, not two overlapping ones
    assert candidate.raw == raw
    normalized = normalize_quantity(candidate.raw)
    assert normalized.normalized_value == value
    assert normalized.modifier == modifier


@pytest.mark.parametrize("text, raw, value", [
    ("The death toll has risen to 31.", "31", 31),
    ("Floods affected 58 lakh people.", "58 lakh", 5_800_000),
    ("There were four more deaths in Cumilla.", "four", 4),
])
def test_number_without_a_modifier_is_unchanged(text, raw, value):
    [candidate] = quantities(text)
    assert candidate.raw == raw
    normalized = normalize_quantity(candidate.raw)
    assert normalized.normalized_value == value
    assert normalized.modifier is None


def test_offsets_point_at_the_whole_phrase():
    text = "Officials said at least 58 lakh people have been affected across 11 districts."
    first = quantities(text)[0]
    assert text[first.start_char:first.end_char] == "at least 58 lakh"


def test_modifier_word_inside_another_word_is_not_taken():
    # "some" is a modifier, "awesome" is not
    [candidate] = quantities("The awesome three rescuers saved everyone.")
    assert candidate.raw == "three"
