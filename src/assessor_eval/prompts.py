"""Assessor prompts: instructions + rubric + task context, for the essay and CV domains.

Everything here is identical for every essay, so it forms a stable (cacheable) system-prompt
prefix. It contains no per-essay information; the essay goes in the user message.
"""

import json
import re
from functools import cache
from pathlib import Path

ASAP = Path(__file__).resolve().parents[2] / "data" / "asap"
CV = Path(__file__).resolve().parents[2] / "data" / "cv"
MAX_SCORE = {"essay": 6, "cv": 5}

OUTPUT_FORMAT = """Respond in exactly this format, starting with the score:
SCORE: <a single whole number from 1 to 6>
RATIONALE: <2-4 sentences explaining the score with reference to the rubric>"""

CV_OUTPUT_FORMAT = """Respond in exactly this format, starting with the score:
SCORE: <a single whole number from 1 to 5>
RATIONALE: <2-4 sentences explaining the score with reference to the requirements and the fit rubric>"""


@cache
def rubric() -> str:
    """The rubric text without the attribution comment at the top of rubric.md."""
    return (ASAP / "rubric.md").read_text().split("-->", 1)[1].strip()


@cache
def assignment() -> dict:
    return json.loads((ASAP / "task.json").read_text())


@cache
def job_ad() -> str:
    """The fictional job ad with its fit rubric, minus the italic provenance note (not shown to the model)."""
    lines = (CV / "job_ad.md").read_text().splitlines()
    text = "\n".join(l for l in lines if not l.startswith("*Example Co. is a fictional"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@cache
def system_prompt(domain: str = "essay") -> str:
    if domain == "cv":
        return f"""You are an experienced technical recruiter. You will assess how well one candidate fits the job \
below, using their profile from a job platform: position, stated experience, English level and CV.

<job_ad>
{job_ad()}
</job_ad>

Score the candidate's overall fit on a scale of 1 (minimum) to 5 (maximum) using the fit rubric in the job ad.

{CV_OUTPUT_FORMAT}"""
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


def user_prompt(text: str, domain: str = "essay") -> str:
    return f"Score this {'candidate' if domain == 'cv' else 'essay'}.\n\n{text}"
