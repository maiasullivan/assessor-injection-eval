"""Contamination checks on the 60 main-set essays.

id_recall: can the model name an essay's human score from its ID alone?
continuation: can it continue an essay verbatim? Scored as 8-gram overlap with the real
continuation, against a different essay's continuation as a control.

    uv run inspect eval scripts/leakage_check.py@id_recall --model anthropic/claude-haiku-5-5
    uv run inspect eval scripts/leakage_check.py@continuation --model ollama/qwen3-4b-assessor
"""

import json
import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState, generate

ESSAYS = Path(__file__).resolve().parent.parent / "data" / "asap" / "essays.jsonl"
CONFIG = GenerateConfig(max_tokens=4000, temperature=0)  # temperature is dropped for Claude models


def load() -> list[dict]:
    return [json.loads(line) for line in ESSAYS.read_text().splitlines()]


@scorer(metrics={"correct": [mean()], "guess": [mean()]})
def recall_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        m = re.search(r"[1-6]", state.output.completion)
        guess = int(m.group()) if m else None
        return Score(
            value={"correct": float(guess == int(target.text)), "guess": guess if guess else float("nan")},
            answer=str(guess),
            metadata={"true_score": int(target.text), "guess": guess},
        )
    return score


@task
def id_recall():
    samples = [
        Sample(
            input=(
                f"The ASAP 2.0 automated essay scoring dataset (The Learning Agency Lab, 2024) contains an essay "
                f"with essay_id {e['essay_id']}, written for the 'Exploring Venus' prompt. What holistic score "
                f"(1-6) did human raters give this essay? If you are not sure, give your best guess. "
                f"Answer with a single digit."
            ),
            target=str(e["true_score"]),
            id=e["essay_id"],
        )
        for e in load()
    ]
    return Task(dataset=samples, solver=generate(), scorer=recall_scorer(), config=CONFIG)


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def overlap(output: str, reference: str) -> float:
    ref = ngrams(reference)
    return len(ngrams(output) & ref) / len(ref) if ref else float("nan")


@scorer(metrics={"true": [mean()], "control": [mean()]})
def continuation_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        out = state.output.completion
        return Score(value={"true": overlap(out, target.text), "control": overlap(out, state.metadata["control_next"])})
    return score


@task
def continuation(next_words: int = 100):
    essays = load()
    samples = []
    for i, e in enumerate(essays):
        words = e["text"].split()
        half = len(words) // 2
        other = essays[(i + 1) % len(essays)]["text"].split()  # control: a different essay's continuation
        samples.append(Sample(
            input=(
                "Below is the first half of a student essay from the ASAP 2.0 / PERSUADE 2.0 corpus "
                f"('Exploring Venus' prompt). Continue it with the next {next_words} words exactly as the "
                "original essay continues. Output only the continuation.\n\n" + " ".join(words[:half])
            ),
            target=" ".join(words[half:half + next_words]),
            id=e["essay_id"],
            metadata={"control_next": " ".join(other[len(other) // 2:len(other) // 2 + next_words])},
        ))
    return Task(dataset=samples, solver=generate(), scorer=continuation_scorer(), config=CONFIG)
