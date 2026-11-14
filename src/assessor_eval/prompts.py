"""Assessor prompt: instructions + rubric + assignment + source article.

Everything here is identical for every essay, so it forms a stable (cacheable) system-prompt
prefix. It contains no per-essay information; the essay goes in the user message.
"""

import json
from functools import cache
from pathlib import Path

ASAP = Path(__file__).resolve().parents[2] / "data" / "asap"

OUTPUT_FORMAT = """Respond in exactly this format, starting with the score:
SCORE: <a single whole number from 1 to 6>
RATIONALE: <2-4 sentences explaining the score with reference to the rubric>"""


@cache
def rubric() -> str:
    """The rubric text without the attribution comment at the top of rubric.md."""
    return (ASAP / "rubric.md").read_text().split("-->", 1)[1].strip()


@cache
def assignment() -> dict:
    return json.loads((ASAP / "task.json").read_text())


@cache
def system_prompt() -> str:
    task = assignment()
    sources = "\n\n".join(task["source_texts"])
    return f"""You are an experienced rater of student writing. You will assign a holistic score to one \
essay written by a US student in grades 6-12 in response to the assignment below. Students had the \
source article to refer to while writing.

<assignment>
{task["assignment"]}
</assignment>

<source_article>
{sources}
</source_article>

Score the essay on a scale of 1 (minimum) to 6 (maximum) using this holistic rubric:

<rubric>
{rubric()}
</rubric>

{OUTPUT_FORMAT}"""


def user_prompt(essay: str) -> str:
    return f"Score this essay.\n\n{essay}"
