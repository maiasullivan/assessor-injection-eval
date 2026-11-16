"""Part A analysis: agreement with the Ollama runs, shift vs flatten, entropy as a detector.

    uv run --group whitebox python whitebox/analyse_distribution.py

Writes results/whitebox/: agreement.csv, weak_summary.csv, auroc.csv, fig3_expected_score.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from analyse import LOGS, load  # noqa: E402
from figstyle import ATTACK_LABELS, COLORS, DEFENCE_LABELS, GRID, INK, MUTED, PAPER, RULE, WIDTH, panel, save, style  # noqa: E402

OUT = ROOT / "results" / "whitebox"
QWEN = COLORS["qwen"]
N_BOOT, RNG = 2000, np.random.default_rng(0)
ATTACKS = ["direct", "fake_note", "keyword"]


def agreement(dist: pd.DataFrame) -> pd.DataFrame:
    """Join each HF row to the Ollama generation for the same essay and condition."""
    gen = pd.concat([load(["qwen"]).assign(set="main"), load(["qwen"], LOGS.parent / "variants").assign(set="variants")])
    gen = gen.dropna(subset=["pred"])
    keys = ["set", "attack", "defence", "essay_id"]
    main = dist[dist.set == "main"].merge(gen[gen.set == "main"][keys + ["pred"]], on=keys)
    var = dist[dist.set == "variants"].merge(gen[gen.set == "variants"][keys + ["variant", "pred"]], on=keys + ["variant"])
    both = pd.concat([main, var])
    both["agree"] = both.argmax == both.pred.astype(int)
    return both


def weak_summary(dist: pd.DataFrame) -> pd.DataFrame:
    w = dist[dist.tier == "weak"]
    cols = ["expected", "p_pass", "entropy", "max_prob"]
    clean = w[(w.set == "main") & (w.attack == "none")].groupby("defence")[cols].mean().add_suffix("_clean")
    by_var = w[w.set == "variants"].groupby(["defence", "attack", "variant"])[cols].mean().reset_index().assign(set="variants")
    main = w[w.set == "main"].groupby(["defence", "attack"])[cols].mean().reset_index().assign(set="main", variant=-1)
    out = pd.concat([main, by_var]).merge(clean, left_on="defence", right_index=True)
    for c in ("expected", "entropy"):
        out[f"d_{c}"] = out[c] - out[f"{c}_clean"]
    return out[["set", "defence", "attack", "variant", *cols, "d_expected", "d_entropy"]]


def auroc(dist: pd.DataFrame) -> pd.DataFrame:
    """Attacked (positive) vs clean (negative) weak essays under the same defence; bootstrap over essays."""
    w = dist[dist.tier == "weak"]
    clean = w[(w.set == "main") & (w.attack == "none")].set_index(["defence", "essay_id"])
    out = []
    groups = [("main", d, a, -1, g) for (d, a), g in w[(w.set == "main") & (w.attack != "none")].groupby(["defence", "attack"])]
    groups += [("variants", d, a, v, g) for (d, a, v), g in w[w.set == "variants"].groupby(["defence", "attack", "variant"])]
    for set_, d, a, v, g in groups:
        g = g.set_index("essay_id")
        c = clean.loc[d].loc[g.index]
        row = {"set": set_, "defence": d, "attack": a, "variant": v, "n": len(g)}
        for name, att, cl in [("entropy", g.entropy, c.entropy), ("uncertainty", 1 - g.max_prob, 1 - c.max_prob)]:
            att, cl = att.to_numpy(), cl.to_numpy()
            y = np.r_[np.ones(len(att)), np.zeros(len(cl))]
            row[f"auroc_{name}"] = roc_auc_score(y, np.r_[att, cl])
            boots = []
            for _ in range(N_BOOT):
                i = RNG.integers(0, len(att), len(att))  # resample essays, keeping each essay's pair
                if len(set(np.r_[att[i], cl[i]])) > 1:
                    boots.append(roc_auc_score(y, np.r_[att[i], cl[i]]))
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = np.percentile(boots, [2.5, 97.5])
        out.append(row)
    return pd.DataFrame(out)


def fig_expected(dist: pd.DataFrame) -> None:
    w = dist[(dist.tier == "weak") & (dist.set == "main")]
    conds = ["none", *ATTACKS]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
    for ax, letter, d in zip(axes, "ab", ["none", "spotlight"]):
        piv = w[w.defence == d].pivot(index="essay_id", columns="attack", values="expected")[conds]
        x = np.arange(len(conds))
        for _, r in piv.iterrows():
            for j in range(1, len(conds)):  # line from each essay's clean score to its attacked score
                ax.plot([0, j], [r["none"], r[conds[j]]], color=GRID, lw=0.6, zorder=1)
        for j, c in enumerate(conds):
            jitter = RNG.uniform(-0.08, 0.08, len(piv))
            ax.scatter(x[j] + jitter, piv[c], s=12, color=QWEN, edgecolor=PAPER, linewidth=0.5, zorder=2)
            ax.hlines(piv[c].mean(), x[j] - 0.22, x[j] + 0.22, color=INK, lw=1.4, zorder=3)
        ax.axhline(4, color=RULE, lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.set_xticks(x, [ATTACK_LABELS[c] for c in conds])
        ax.tick_params(axis="x", length=0)
        panel(ax, letter, DEFENCE_LABELS[d])
        ax.set_ylim(0.8, 6.2)
        style(ax, grid=None)
    axes[0].set_ylabel("Expected score $\\mathrm{E}[s]$")
    axes[1].text(len(conds) - 0.55, 4.08, "pass", fontsize=8, color=MUTED, ha="right", va="bottom", fontstyle="italic")
    fig.tight_layout()
    save(fig, OUT / "fig3_expected_score.png")


def main() -> None:
    dist = pd.read_csv(OUT / "score_dist.csv")
    print(f"digit mass: min {dist.digit_mass.min():.4f}, median {dist.digit_mass.median():.4f}")

    ag = agreement(dist)
    ag_table = ag.groupby(["set", "defence", "attack"]).agg(n=("agree", "size"), agree=("agree", "mean")).reset_index()
    ws, au = weak_summary(dist), auroc(dist)
    for name, t in [("agreement", ag_table), ("weak_summary", ws), ("auroc", au)]:
        t.round(3).to_csv(OUT / f"{name}.csv", index=False)
    fig_expected(dist)

    pd.set_option("display.width", 180)
    print(f"\nArgmax agreement with Ollama generation: {ag.agree.mean():.2f} (n = {len(ag)}); "
          f"mean |E[s] - generated| = {(ag.expected - ag.pred).abs().mean():.2f}")
    print(ag_table.round(2).to_string(index=False))
    print("\nWeak essays\n", ws.round(2).to_string(index=False))
    print("\nAUROC (attacked vs clean weak essays)\n", au.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
