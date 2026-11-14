"""Inspect task: an LLM assessor scores ASAP 2.0 essays, optionally attacked and/or defended.

    uv run inspect eval src/assessor_eval/task.py -T split=dev --model anthropic/claude-haiku-5-5
    uv run inspect eval src/assessor_eval/task.py -T attack=keyword -T defence=spotlight --model ...

The system and user messages are built literally here rather than with Inspect's
system_message() solver, which str.format()s sample metadata into its template: that would be
one stray "{true_score}" away from leaking the label. Score and id go only in target/metadata.
"""

import json

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.solver import generate

from assessor_eval.attacks import inject
from assessor_eval.defences import spotlight
from assessor_eval.prompts import ASAP, system_prompt, user_prompt
from assessor_eval.scorer import assessor_scorer

SPLITS = {"main": "essays.jsonl", "dev": "dev.jsonl"}
DEFENCES = ("none", "spotlight")


def build_sample(essay: dict, attack: str, defence: str, position: str) -> Sample:
    text = inject(essay["text"], attack, key=essay["essay_id"], position=position)
    system = system_prompt()
    if defence == "spotlight":
        addendum, text = spotlight(text)
        system = f"{system}\n\n{addendum}"
    elif defence != "none":
        raise ValueError(f"unknown defence {defence!r}; expected one of {DEFENCES}")
    return Sample(
        input=[ChatMessageSystem(content=system), ChatMessageUser(content=user_prompt(text))],
        target=str(essay["true_score"]),
        id=essay["essay_id"],
        metadata={
            "true_score": essay["true_score"],
            "tier": essay["tier"],
            "words": essay["words"],
            "attack": attack,
            "defence": defence,
            "position": position,
        },
    )


def load_split(split: str) -> list[dict]:
    return [json.loads(line) for line in (ASAP / SPLITS[split]).read_text().splitlines()]


@task
def assessor(attack: str = "none", defence: str = "none", position: str = "end", split: str = "main"):
    samples = [build_sample(e, attack, defence, position) for e in load_split(split)]
    return Task(
        dataset=samples,
        solver=generate(),
        scorer=assessor_scorer(),
        # temperature is dropped for Claude models (default sampling + epochs); it makes Ollama deterministic
        config=GenerateConfig(max_tokens=4000, temperature=0),
        metadata={"attack": attack, "defence": defence, "position": position, "split": split},
    )
