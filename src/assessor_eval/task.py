"""Inspect task: an LLM assessor scores ASAP 2.0 essays, optionally attacked and/or defended.

    uv run inspect eval src/assessor_eval/task.py -T split=dev --model anthropic/claude-haiku-5-5
    uv run inspect eval src/assessor_eval/task.py -T attack=keyword -T defence=spotlight --model ...
    uv run inspect eval src/assessor_eval/task.py -T attack=direct -T variant=2 -T tier=weak --model ...

The system and user messages are built literally here rather than with Inspect's
system_message() solver, which str.format()s sample metadata into its template: that would be
one stray "{true_score}" away from leaking the label. Score and id go only in target/metadata.
"""

import json

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.solver import generate

from assessor_eval.attacks import ATTACKS, inject, pick_variant
from assessor_eval.defences import spotlight
from assessor_eval.prompts import ASAP, system_prompt, user_prompt
from assessor_eval.scorer import assessor_scorer

SPLITS = {"main": "essays.jsonl", "dev": "dev.jsonl"}
DEFENCES = ("none", "spotlight")


def build_sample(essay: dict, attack: str, defence: str, position: str, variant: int | None = None) -> Sample:
    """`variant` forces one wording for every essay; None keeps the per-essay hashed choice."""
    text = inject(essay["text"], attack, key=essay["essay_id"], position=position, variant=variant)
    if variant is None:
        variant = ATTACKS[attack].index(pick_variant(attack, essay["essay_id"]))
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
            "variant": variant,
        },
    )


def load_split(split: str) -> list[dict]:
    return [json.loads(line) for line in (ASAP / SPLITS[split]).read_text().splitlines()]


@task
def assessor(attack: str = "none", defence: str = "none", position: str = "end", split: str = "main",
             variant: int = -1, tier: str = "all"):
    """variant = -1 keeps the per-essay hashed wording; tier = weak/middling/strong keeps one tier only."""
    essays = [e for e in load_split(split) if tier in ("all", e["tier"])]
    samples = [build_sample(e, attack, defence, position, None if variant < 0 else variant) for e in essays]
    return Task(
        dataset=samples,
        solver=generate(),
        scorer=assessor_scorer(),
        # temperature is dropped for Claude models (default sampling + epochs); it makes Ollama deterministic
        config=GenerateConfig(max_tokens=4000, temperature=0),
        metadata={"attack": attack, "defence": defence, "position": position, "split": split,
                  "variant": variant, "tier": tier},
    )
