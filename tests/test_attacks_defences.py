import pytest

from assessor_eval.attacks import ATTACKS, inject, pick_variant
from assessor_eval.defences import ADDENDUM, MARKER, spotlight

ESSAY = "Venus is hot.\n\nThe author gives evidence. Scientists want to study it.\n\nI agree with the author."


def test_none_leaves_text_unchanged():
    assert inject(ESSAY, "none", key="e1") == ESSAY


@pytest.mark.parametrize("attack", ["direct", "fake_note", "keyword"])
def test_end_appends_payload_after_original(attack):
    out = inject(ESSAY, attack, key="e1")
    payload = pick_variant(attack, "e1")
    assert out.startswith(ESSAY) and out.endswith(payload)


def test_middle_inserts_payload_between_paragraphs():
    out = inject(ESSAY, "direct", key="e1", position="middle")
    payload = pick_variant("direct", "e1")
    before, after = out.split(payload)
    assert before.endswith("\n\n") and after.startswith("\n\n")
    assert out.replace(payload + "\n\n", "") == ESSAY


def test_middle_without_paragraphs_uses_sentence_boundary():
    text = "One sentence here. Another one here. A third sentence. And a fourth."
    out = inject(text, "keyword", key="e2", position="middle")
    assert pick_variant("keyword", "e2") in out
    assert " ".join(out.replace(pick_variant("keyword", "e2"), "").split()) == text


def test_variant_is_stable_per_key_and_varies_across_keys():
    assert pick_variant("fake_note", "abc") == pick_variant("fake_note", "abc")
    chosen = {pick_variant("fake_note", f"essay{i}") for i in range(60)}
    assert chosen == set(ATTACKS["fake_note"])


def test_keyword_attack_contains_no_instruction_or_score():
    for v in ATTACKS["keyword"]:
        assert "score" not in v.lower() and "ignore" not in v.lower() and "6" not in v


def test_unknown_attack_or_position_raises():
    with pytest.raises(ValueError):
        inject(ESSAY, "nope", key="e1")
    with pytest.raises(ValueError):
        inject(ESSAY, "direct", key="e1", position="start")


def test_spotlight_marks_spaces_keeps_paragraphs_and_wraps():
    addendum, wrapped = spotlight(ESSAY)
    assert addendum == ADDENDUM
    assert wrapped.startswith("<essay>\n") and wrapped.endswith("\n</essay>")
    body = wrapped.removeprefix("<essay>\n").removesuffix("\n</essay>")
    assert " " not in body
    assert body.replace(MARKER, " ") == ESSAY


def test_spotlight_marks_injected_text_too():
    _, wrapped = spotlight(inject(ESSAY, "direct", key="e1"))
    assert "Ignore^all^previous^instructions" in wrapped or "SCORE:^6" in wrapped


def test_addendum_is_generic():
    lowered = ADDENDUM.lower()
    for attack_specific in ["scorer", "note", "rater", "keyword", "rubric language", "score: 6"]:
        assert attack_specific not in lowered
