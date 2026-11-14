"""Analyse the wording follow-up (scripts/run_variants.py): ASR per wording on all 20 weak essays.

    uv run python scripts/analyse_variants.py

Writes results/variants_asr.csv (and prints it). The clean pass rate of each essay comes from the
main matrix (same model, defence, essays and epochs), so ASR = attacked - clean as in asr.csv.
"""

import numpy as np
import pandas as pd

from analyse import DISCLOSED, LOGS, MODELS, OUT, PASS_SCORE, boot_ci, load, per_essay_means

VLOGS = LOGS.parent / "variants"


def main() -> None:
    models = [m for m in MODELS if (VLOGS / m).exists()]
    att = load(models, VLOGS)
    clean = per_essay_means(load(models)).query("attack == 'none' and tier == 'weak'")
    clean = clean[["model", "defence", "essay_id", "passed"]]
    att["disclosed"] = att.reply.str.contains(DISCLOSED, case=False) & (att.pred >= PASS_SCORE)

    e = (att.groupby(["model", "defence", "attack", "variant", "essay_id"], as_index=False)
            .agg(passed=("pred", lambda x: (x.dropna() >= PASS_SCORE).mean()), pred=("pred", "mean"),
                 disclosed=("disclosed", "sum"), hits=("pred", lambda x: int((x.dropna() >= PASS_SCORE).sum()))))
    e = e.merge(clean, on=["model", "defence", "essay_id"], suffixes=("", "_clean"))

    out = []
    for (m, d, a, v), g in e.groupby(["model", "defence", "attack", "variant"]):
        diff = (g.passed - g.passed_clean).to_numpy(float)
        lo, hi = boot_ci(diff)
        out.append({"model": m, "defence": d, "attack": a, "variant": v, "n_weak": len(g),
                    "attacked_rate": g.passed.mean(), "clean_rate": g.passed_clean.mean(), "asr": diff.mean(),
                    "ci_low": lo, "ci_high": hi, "essays_flipped": int((g.passed > 0).sum()),
                    "disclosed": f"{int(g.disclosed.sum())}/{int(g.hits.sum())}"})
    table = pd.DataFrame(out)
    table.round(3).to_csv(OUT / "variants_asr.csv", index=False)
    pd.set_option("display.width", 160)
    print(table.round(2).to_string(index=False))

    # Are the same essays vulnerable across wordings? Count, per essay, how many wordings flipped it.
    per = e[e.passed > 0].groupby(["model", "defence", "attack"]).essay_id.value_counts()
    print("\nEssays flipped by k of 3 wordings\n", per.groupby(level=[0, 1, 2]).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
