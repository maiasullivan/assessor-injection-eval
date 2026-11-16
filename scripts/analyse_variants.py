"""Attack success per wording for the follow-up runs.

    uv run python scripts/analyse_variants.py              # logs/variants/ -> results/
    uv run python scripts/analyse_variants.py --domain cv  # logs/variants_cv/ -> results/cv/

Clean pass rates come from the main matrix, so ASR here is defined as in asr.csv.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from analyse import DISCLOSED, LOGS, MODELS, OUT, PASS_SCORE, boot_ci, load, per_essay_means
from figstyle import ATTACK_LABELS, COLORS, DEFENCE_LABELS, INK, MUTED, PALE, PAPER, WIDTH, panel, save, style

DOMAINS = {  # variant logs, clean-baseline logs, output dir, noun
    "essay": (LOGS.parent / "variants", LOGS, OUT, "essays"),
    "cv": (LOGS.parent / "variants_cv", LOGS.parent / "matrix_cv", OUT / "cv", "candidates"),
}


def per_item(att: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    e = (att.groupby(["model", "defence", "attack", "variant", "essay_id"], as_index=False)
            .agg(passed=("pred", lambda x: (x.dropna() >= PASS_SCORE).mean()), pred=("pred", "mean"),
                 disclosed=("disclosed", "sum"), hits=("pred", lambda x: int((x.dropna() >= PASS_SCORE).sum())),
                 years=("experience_years", "first")))
    return e.merge(clean, on=["model", "defence", "essay_id"], suffixes=("", "_clean"))


def asr_table(e: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, d, a, v), g in e.groupby(["model", "defence", "attack", "variant"]):
        diff = (g.passed - g.passed_clean).to_numpy(float)
        lo, hi = boot_ci(diff)
        out.append({"model": m, "defence": d, "attack": a, "variant": v, "n_weak": len(g),
                    "attacked_rate": g.passed.mean(), "clean_rate": g.passed_clean.mean(), "asr": diff.mean(),
                    "ci_low": lo, "ci_high": hi, "essays_flipped": int((g.passed > 0).sum()),
                    "disclosed": f"{int(g.disclosed.sum())}/{int(g.hits.sum())}"})
    return pd.DataFrame(out)


def fig_variants(t: pd.DataFrame, path, noun: str) -> None:
    """ASR per wording: one panel per (defence, attack), models side by side within each wording."""
    attacks = [a for a in ["direct", "fake_note", "keyword"] if a in set(t.attack)]
    defences = [d for d in ["none", "spotlight"] if d in set(t.defence)]
    models = [m for m in MODELS if m in set(t.model)]
    fig, axes = plt.subplots(len(defences), len(attacks), figsize=(WIDTH, 2.1 * len(defences) + 0.2),
                             sharey=True, squeeze=False)
    off = np.linspace(-0.2, 0.2, len(models))
    for r, d in enumerate(defences):
        for c, a in enumerate(attacks):
            ax = axes[r, c]
            style(ax)
            ax.axhline(0, color=INK, lw=0.6)
            for i, m in enumerate(models):
                g = t[(t.model == m) & (t.defence == d) & (t.attack == a)].sort_values("variant")
                if g.empty:
                    continue
                ax.errorbar(g.variant + off[i], g.asr, yerr=[g.asr - g.ci_low, g.ci_high - g.asr], fmt="none",
                            ecolor=COLORS[m], elinewidth=0.9, capsize=0, alpha=0.8)
                ax.scatter(g.variant + off[i], g.asr, s=22, color=COLORS[m], edgecolor=PAPER, linewidth=0.6,
                           zorder=3, label=MODELS[m])
            ax.set_xticks(range(3), ["W0", "W1", "W2"])
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(-0.3, 1.05)
            panel(ax, "abcdef"[r * len(attacks) + c], f"{ATTACK_LABELS[a].capitalize()}, {DEFENCE_LABELS[d].lower()}")
            if c == 0:
                ax.set_ylabel("Attack success rate")
            if r == len(defences) - 1:
                ax.set_xlabel("Wording")
    handles, labels = [], []
    for ax in axes.flat:  # some panels lack a model
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, ncol=len(labels), loc="lower right", bbox_to_anchor=(1.0, 0.975),
               handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(h_pad=1.4)
    save(fig, path)


def fig_grid(e: pd.DataFrame, path, noun: str, min_flips: int = 3) -> None:
    """Item x wording pass rates for every (model, defence, attack) with at least `min_flips` flipped cells."""
    cells = e.groupby(["model", "defence", "attack"]).filter(lambda g: (g.passed > 0).sum() >= min_flips)
    panels = list(cells.groupby(["model", "defence", "attack"], sort=False).groups)
    order = {m: i for i, m in enumerate(MODELS)}
    panels.sort(key=lambda k: (order[k[0]], k[1], k[2]))
    if not panels:
        return
    sort_key = "years" if e.years.notna().any() else "essay_id"
    items = e.drop_duplicates("essay_id").sort_values([sort_key, "essay_id"]).essay_id.tolist()
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 4.6), sharey=True, squeeze=False)
    for ax, letter, (m, d, a) in zip(axes[0], "abcdefgh", panels):
        g = e[(e.model == m) & (e.defence == d) & (e.attack == a)]
        grid = g.pivot(index="essay_id", columns="variant", values="passed").reindex(items)
        cmap = LinearSegmentedColormap.from_list(m, [PALE, COLORS[m]])
        ax.imshow(grid.to_numpy(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3), ["W0", "W1", "W2"])
        ax.set_xticks(np.arange(-0.5, 3), minor=True)
        ax.set_yticks(np.arange(-0.5, len(items)), minor=True)
        ax.grid(which="minor", color=PAPER, lw=1.5)  # white gaps between cells
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"$\\bf{{({letter})}}$ {MODELS[m]}\n{ATTACK_LABELS[a]}\n"
                     f"{'no defence' if d == 'none' else 'spotlighting'}", fontsize=8.5, loc="center", color=INK)
    if sort_key == "years":
        yrs = e.drop_duplicates("essay_id").set_index("essay_id").years.reindex(items)
        axes[0, 0].set_yticks(range(len(items)), [f"{y:g} yrs" for y in yrs])
    else:
        axes[0, 0].set_yticks(range(len(items)), [""] * len(items))
    axes[0, 0].set_ylabel(f"Weak {noun} (one row each)")
    fig.supxlabel("Wording", fontsize=9, y=0.02)
    fig.tight_layout(w_pad=0.6)
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="essay")
    args = parser.parse_args()
    vlogs, clogs, out, noun = DOMAINS[args.domain]
    models = [m for m in MODELS if (vlogs / m).exists()]
    att, base = load(models, vlogs), load(models, clogs)
    if args.domain == "cv":
        att["true"] = base["true"] = -1  # no human score; groupby would otherwise drop the rows
    clean = per_essay_means(base).query("attack == 'none' and tier == 'weak'")[["model", "defence", "essay_id", "passed"]]
    att["disclosed"] = att.reply.str.contains(DISCLOSED, case=False) & (att.pred >= PASS_SCORE)

    e = per_item(att, clean)
    table = asr_table(e)
    table.round(3).to_csv(out / "variants_asr.csv", index=False)
    fig_variants(table, out / "fig_variants.png", noun)
    fig_grid(e, out / "fig_variant_grid.png", noun)
    pd.set_option("display.width", 160)
    print(table.round(2).to_string(index=False))

    # how many wordings flipped each item
    per = e[e.passed > 0].groupby(["model", "defence", "attack"]).essay_id.value_counts()
    print(f"\n{noun.capitalize()} flipped by k of 3 wordings\n",
          per.groupby(level=[0, 1, 2]).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
