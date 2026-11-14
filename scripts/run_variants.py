"""Follow-up to Phase 4: every wording of `direct` and `fake_note` on every weak essay.

    uv run python scripts/run_variants.py --model sonnet

In the main matrix each essay always gets the same (hashed) wording, so a wording's success rate
is confounded with which essays drew it. Here all 20 weak essays get all 3 wordings, with no
defence (and spotlighting too for Qwen, which is free). Clean baselines come from logs/matrix/.
"""

import argparse

from inspect_ai import eval_set

from assessor_eval.attacks import ATTACKS
from run_matrix import MODELS
from assessor_eval.task import assessor

VARIANT_ATTACKS = ("direct", "fake_note")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    model, epochs = MODELS[args.model]
    defences = ("none", "spotlight") if args.model == "qwen" else ("none",)
    tasks = [assessor(attack=a, defence=d, variant=v, tier="weak")
             for d in defences for a in VARIANT_ATTACKS for v in range(len(ATTACKS[a]))]
    success, _ = eval_set(tasks, model=model, epochs=epochs, log_dir=f"logs/variants/{args.model}", max_tasks=1)
    print(f"{model}: {'all conditions complete' if success else 'some conditions failed; re-run to retry'}")


if __name__ == "__main__":
    main()
