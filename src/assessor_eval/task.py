"""Inspect task: an LLM assessor scores ASAP 2.0 essays, optionally attacked and/or defended.

    uv run inspect eval src/assessor_eval/task.py -T split=dev --model anthropic/claude-haiku-5-5
    uv run inspect eval src/assessor_eval/task.py -T attack=keyword -T defence=spotlight --model ...
    uv run inspect eval src/assessor_eval/task.py -T attack=direct -T variant=2 -T tier=weak --model ...
    uv run inspect eval src/assessor_eval/task.py -T domain=cv -T attack=keyword --model ...

domain="cv" screens Djinni candidate profiles against the fictional job ad (data/cv/). There is no
human score: the target is "proxy" and the tier (from stated experience years) is analysis-only.

The system and user messages are built literally here rather than with Inspect's
system_message() solver, which str.format()s sample metadata into its template: that would be
one stray "{true_score}" away from leaking the label. Score and id go only in target/metadata.
"""

import json

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.solver import generate

from assessor_eval.attacks import ATTACK_SETS, inject, pick_variant
from assessor_eval.defences import spotlight
from assessor_eval.prompts import ASAP, CV, MAX_SCORE, system_prompt, user_prompt
from assessor_eval.scorer import assessor_scorer

SPLITS = {"essay": (ASAP, {"main": "essays.jsonl", "dev": "dev.jsonl"}),
          "cv": (CV, {"main": "candidates.jsonl", "dev": "candidates_dev.jsonl"})}
DEFENCES = ("none", "spotlight")


def build_sample(item: dict, attack: str, defence: str, position: str, variant: int | None = None,
                 domain: str = "essay") -> Sample:
    """`variant` forces one wording for every item; None keeps the per-item hashed choice."""
    key = item["cand_id"] if domain == "cv" else item["essay_id"]
    text = inject(item["text"], attack, key=key, position=position, variant=variant, domain=domain)
    if variant is None:
        variant = ATTACK_SETS[domain][attack].index(pick_variant(attack, key, domain=domain))
    system = system_prompt(domain)
    if defence == "spotlight":
        addendum, text = spotlight(text, domain)
        system = f"{system}\n\n{addendum}"
    elif defence != "none":
        raise ValueError(f"unknown defence {defence!r}; expected one of {DEFENCES}")
    meta = {"tier": item["tier"], "words": item["words"], "attack": attack, "defence": defence,
            "position": position, "variant": variant, "domain": domain, "max_score": MAX_SCORE[domain]}
    if domain == "cv":
        meta |= {"true_score": None, "experience_years": item["experience_years"]}
        target = "proxy"
    else:
        meta["true_score"] = item["true_score"]
        target = str(item["true_score"])
    return Sample(
        input=[ChatMessageSystem(content=system), ChatMessageUser(content=user_prompt(text, domain))],
        target=target,
        id=key,
        metadata=meta,
    )


def load_split(split: str, domain: str = "essay") -> list[dict]:
    folder, files = SPLITS[domain]
    return [json.loads(line) for line in (folder / files[split]).read_text().splitlines()]


@task
def assessor(attack: str = "none", defence: str = "none", position: str = "end", split: str = "main",
             variant: int = -1, tier: str = "all", domain: str = "essay"):
    """variant = -1 keeps the per-item hashed wording; tier = weak/middling/strong keeps one tier only."""
    items = [e for e in load_split(split, domain) if tier in ("all", e["tier"])]
    samples = [build_sample(e, attack, defence, position, None if variant < 0 else variant, domain) for e in items]
    return Task(
        dataset=samples,
        solver=generate(),
        scorer=assessor_scorer(),
        # temperature is dropped for Claude models (default sampling + epochs); it makes Ollama deterministic
        config=GenerateConfig(max_tokens=4000, temperature=0),
        metadata={"attack": attack, "defence": defence, "position": position, "split": split,
                  "variant": variant, "tier": tier, "domain": domain},
    )
