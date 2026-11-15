"""Phase 4: run every attack x defence condition for each model on the 60-essay main set.

    uv run python scripts/run_matrix.py --model haiku
    uv run python scripts/run_matrix.py --model sonnet
    uv run python scripts/run_matrix.py --model qwen
    uv run python scripts/run_matrix.py --model haiku --domain cv            # CV screening (logs/matrix_cv/)
    uv run python scripts/run_matrix.py --model haiku --domain cv --split dev # quick check (logs/dev_cv/)

Each model gets its own log dir under logs/matrix/. eval_set skips conditions that already
finished, so a re-run after an interruption only does what's missing. Conditions run one at a
time (max_tasks=1) so the cached system prompt is reused within a condition.
"""

import argparse

from inspect_ai import eval_set

from assessor_eval.attacks import ATTACKS
from assessor_eval.task import DEFENCES, assessor

MODELS = {
    "haiku": ("anthropic/claude-haiku-5-5", 3),
    "sonnet": ("anthropic/claude-sonnet-5-5", 2),
    "qwen": ("ollama/qwen3-4b-assessor", 1),  # temperature 0 -> deterministic, 1 epoch
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--domain", choices=("essay", "cv"), default="essay")
    parser.add_argument("--split", choices=("main", "dev"), default="main")
    parser.add_argument("--epochs", type=int, help="override the model's default epochs")
    args = parser.parse_args()
    model, epochs = MODELS[args.model]
    tasks = [assessor(attack=a, defence=d, domain=args.domain, split=args.split) for d in DEFENCES for a in ATTACKS]
    folder = {("essay", "main"): "matrix", ("cv", "main"): "matrix_cv"}.get((args.domain, args.split), f"dev_{args.domain}")
    success, _ = eval_set(tasks, model=model, epochs=args.epochs or epochs, log_dir=f"logs/{folder}/{args.model}", max_tasks=1)
    print(f"{model}: {'all conditions complete' if success else 'some conditions failed; re-run to retry'}")


if __name__ == "__main__":
    main()
