"""The accuracy scorer: matching rule and metrics, on small made-up spans."""
import pytest

from score_extraction import match, prf, qwen_spans, score


def g(start, end, label, text="x"):
    return {"start": start, "end": end, "label": label, "text": text}


def p(start, end, label, kind="QUANTITY", text="x"):
    return {"start": start, "end": end, "label": label, "text": text, "kind": kind}


def claim(start, end, impact, status="ACCEPTED", kind="QUANTITY"):
    return {"char_start": start, "char_end": end, "impact_type": impact, "raw_value": "x",
            "validation_status": status, "candidate_kind": kind}


def test_overlap_with_same_class_is_a_match():
    # Annotated "31 people" (10-19), Qwen found the number "31" (10-12).
    pairs, missed, extra = match([g(10, 19, "FATALITIES")], [p(10, 12, "FATALITIES")])
    assert len(pairs) == 1 and not missed and not extra


def test_wrong_class_is_one_miss_and_one_wrong():
    result = score([{"id": "a", "gold": [g(10, 19, "FATALITIES")],
                     "pred": [p(10, 12, "AFFECTED_PEOPLE")]}])
    assert result["per_class"]["FATALITIES"]["fn"] == 1
    assert result["per_class"]["AFFECTED_PEOPLE"]["fp"] == 1
    assert result["micro"]["tp"] == 0


def test_one_prediction_cannot_match_two_annotations():
    pairs, missed, extra = match([g(0, 5, "DISPLACED"), g(3, 9, "DISPLACED")],
                                 [p(2, 6, "DISPLACED")])
    assert len(pairs) == 1 and len(missed) == 1


def test_numbers_are_matched_before_cue_sentences():
    # The cue sentence (0-100) must not take the annotation that "500" explains.
    gold = [g(20, 23, "RELIEF_INFO")]
    pairs, missed, extra = match(gold, [p(0, 100, "RELIEF_INFO", kind="CUE"),
                                        p(20, 23, "RELIEF_INFO")])
    assert pairs[0][1]["kind"] == "QUANTITY"
    assert len(extra) == 1 and extra[0]["kind"] == "CUE"


def test_several_numbers_inside_one_annotated_phrase_are_not_wrong():
    # Annotated: "1,700 packets of dry food and 650 tons of rice" (0-46) as one RELIEF_INFO.
    result = score([{"id": "a", "gold": [g(0, 46, "RELIEF_INFO")],
                     "pred": [p(0, 5, "RELIEF_INFO"), p(29, 32, "RELIEF_INFO")]}])
    relief = result["per_class"]["RELIEF_INFO"]
    assert (relief["tp"], relief["fp"], relief["inside"]) == (1, 0, 1)
    # A different class inside the phrase is still wrong.
    result = score([{"id": "a", "gold": [g(0, 46, "RELIEF_INFO")],
                     "pred": [p(0, 5, "RELIEF_INFO"), p(29, 32, "FATALITIES")]}])
    assert result["per_class"]["FATALITIES"]["fp"] == 1


def test_class_without_annotations_is_not_measurable_not_zero():
    result = score([{"id": "a", "gold": [g(0, 2, "FATALITIES")], "pred": [p(0, 2, "FATALITIES")]}])
    missing = result["per_class"]["MISSING"]
    assert missing["support"] == 0
    assert missing["recall"] is None and missing["precision"] is None and missing["f1"] is None
    assert result["macro_f1"] == 100.0      # only measurable classes are averaged


def test_wrong_predictions_in_an_unannotated_class_give_zero_precision():
    result = score([{"id": "a", "gold": [], "pred": [p(0, 2, "MISSING")]}])
    assert result["per_class"]["MISSING"]["precision"] == 0.0
    assert result["per_class"]["MISSING"]["recall"] is None


def test_prf_numbers():
    assert prf(8, 2, 2) == pytest.approx((80.0, 80.0, 80.0))
    assert prf(0, 3, 2) == (0.0, 0.0, 0.0)


def test_exact_rule_needs_identical_positions():
    articles = [{"id": "a", "gold": [g(10, 19, "FATALITIES")], "pred": [p(10, 12, "FATALITIES")]}]
    assert score(articles, rule="exact")["micro"]["tp"] == 0
    assert score(articles)["micro"]["tp"] == 1


def test_qwen_spans_keep_only_scored_classes_and_accepted_answers():
    claims = [claim(0, 2, "FATALITIES"),
              claim(5, 7, "RELIEF"),                          # renamed to RELIEF_INFO
              claim(9, 11, "INJURED"),                        # not annotated in this test set
              claim(13, 15, "MISSING", status="REJECTED"),
              claim(17, 19, "DISPLACED", status="REVIEW_REQUIRED"),
              claim(None, None, "FATALITIES")]                # headline: no position
    labels = [s["label"] for s in qwen_spans(claims)]
    assert labels == ["FATALITIES", "RELIEF_INFO"]
    with_review = [s["label"] for s in qwen_spans(claims, ("ACCEPTED", "REVIEW_REQUIRED"))]
    assert with_review == ["FATALITIES", "RELIEF_INFO", "DISPLACED"]


def test_mistake_list_says_what_the_model_said_instead():
    result = score([{"id": "a", "gold": [g(0, 2, "FATALITIES", "31")], "pred": [],
                     "other": [{"start": 0, "end": 2, "label": "INJURED (ACCEPTED)"}]}])
    assert result["errors"][0]["predicted"] == "INJURED (ACCEPTED)"
    assert result["errors"][0]["error"] == "missed"
