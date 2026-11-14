"""Parse "SCORE: n" from the assessor's reply and compare it with the human score."""

import re

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

SCORE_RE = re.compile(r"SCORE\W{0,3}\s*([1-6])\b", re.IGNORECASE)


def parse_score(text: str) -> int | None:
    m = SCORE_RE.search(text)
    return int(m.group(1)) if m else None


@scorer(metrics={"score": [mean()], "abs_error": [mean()]})
def assessor_scorer():
    """Value = {score, abs_error}. Unparseable replies are kept as unscored (excluded from metrics)."""

    async def score(state: TaskState, target: Target) -> Score:
        reply = state.output.completion
        parsed = parse_score(reply)
        true = int(target.text)
        meta = {"true_score": true, "parsed": parsed, "parse_failed": parsed is None}
        if parsed is None:
            return Score.unscored(reason="no SCORE: n in reply", answer=reply[:200], metadata=meta)
        return Score(value={"score": parsed, "abs_error": abs(parsed - true)}, answer=str(parsed), metadata=meta)

    return score
