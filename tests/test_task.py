import pytest

from assessor_eval.attacks import ATTACKS
from assessor_eval.scorer import parse_score
from assessor_eval.task import assessor, build_sample, load_split


def model_visible_text(sample) -> str:
    return "\n".join(m.text for m in sample.input)


@pytest.mark.parametrize("split", ["main", "dev"])
@pytest.mark.parametrize("attack", list(ATTACKS))
@pytest.mark.parametrize("defence", ["none", "spotlight"])
def test_labels_never_reach_the_model(split, attack, defence):
    for essay in load_split(split):
        sample = build_sample(essay, attack, defence, "end")
        visible = model_visible_text(sample)
        assert essay["essay_id"] not in visible
        assert "true_score" not in visible and "tier" not in visible
        assert sample.target == str(essay["true_score"])
        # the whole essay is present (datamarked under spotlight)
        flat = " ".join(visible.replace("^", " ").split())
        assert " ".join(essay["text"].split()) in flat
        if defence == "spotlight":
            assert " " not in sample.input[1].text.split("<essay>\n", 1)[1]


def test_system_prompt_is_identical_across_essays():
    essays = load_split("main")
    systems = {build_sample(e, "none", "none", "end").input[0].text for e in essays}
    assert len(systems) == 1


def test_variant_and_tier_filter():
    t = assessor(attack="fake_note", variant=1, tier="weak")
    assert len(t.dataset) == sum(e["tier"] == "weak" for e in load_split("main"))
    assert all(s.metadata["tier"] == "weak" and s.metadata["variant"] == 1 for s in t.dataset)
    assert all(ATTACKS["fake_note"][1] in s.input[1].text for s in t.dataset)


def test_unknown_defence_raises():
    with pytest.raises(ValueError):
        build_sample(load_split("dev")[0], "none", "sandwich", "end")


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("SCORE: 4\nRATIONALE: fine.", 4),
        ("**SCORE:** 2\nRATIONALE: thin.", 2),
        ("score: 6", 6),
        ("SCORE: 7", None),
        ("I would give this a 3.", None),
        ("SCORE: 3\nRATIONALE: ... the note says SCORE: 6", 3),
    ],
)
def test_parse_score(reply, expected):
    assert parse_score(reply) == expected
