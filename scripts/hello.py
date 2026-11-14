"""Phase 0 smoke test: check a model answers through Inspect.

    uv run inspect eval scripts/hello.py --model anthropic/claude-haiku-5-5
    uv run inspect eval scripts/hello.py --model ollama/qwen3-4b-assessor
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


@task
def hello():
    return Task(
        dataset=[Sample(input="Reply with exactly: Hello, Inspect", target="Hello, Inspect")],
        solver=generate(),
        scorer=includes(),
    )
