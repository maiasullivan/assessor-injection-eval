"""Follow-up to Phase 4: every wording of `direct` and `fake_note` on every weak essay.

    uv run python scripts/run_variants.py --model sonnet
    uv run python scripts/run_variants.py --model sonnet --domain cv   # logs/variants_cv/

In the main matrix each essay always gets the same (hashed) wording, so a wording's success rate
is confounded with which essays drew it. Here all 20 weak essays get all 3 wordings, with no
defence (and spotlighting too for Qwen, which is free). Clean baselines come from logs/matrix/.

CV: all 3 wordings of all three attacks (keyword included: it was the one spotlighting did not stop)
on the 20 weak candidates, with and without spotlighting for every model. Baselines: logs/matrix_cv/.
"""

import argparse

from inspect_ai import eval_set

from assessor_eval.attacks import ATTACK_SETS
from run_matrix import MODELS
from assessor_eval.task import assessor

VARIANT_ATTACKS = ("direct", "fake_note")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--domain", choices=("essay", "cv"), default="essay")
    args = parser.parse_args()
    model, epochs = MODELS[args.model]
    cv = args.domain == "cv"
    defences = ("none", "spotlight") if cv or args.model == "qwen" else ("none",)
    attacks = (*VARIANT_ATTACKS, "keyword") if cv else VARIANT_ATTACKS
    attack_set = ATTACK_SETS[args.domain]
    tasks = [assessor(attack=a, defence=d, variant=v, tier="weak", domain=args.domain)
             for d in defences for a in attacks for v in range(len(attack_set[a]))]
    folder = "variants_cv" if cv else "variants"
    success, _ = eval_set(tasks, model=model, epochs=epochs, log_dir=f"logs/{folder}/{args.model}", max_tasks=1)
    print(f"{model}: {'all conditions complete' if success else 'some conditions failed; re-run to retry'}")


if __name__ == "__main__":
    main()
