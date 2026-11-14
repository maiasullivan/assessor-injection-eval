"""White-box Part B (stretch): does Qwen3-4B "know better" than the score it writes?

    uv run --group whitebox python whitebox/linear_probe.py extract   # forward passes -> .cache/whitebox/
    uv run --group whitebox python whitebox/linear_probe.py analyse   # probes, tables, figure

extract: the residual-stream hidden state at the last prompt token (the position that predicts
the score digit, after the "SCORE: " prefill), at every layer, for
  - probe_set.jsonl: 200 clean, length-matched essays (100 weak, 100 strong), to train the probes
  - the 60-essay main set, clean and under each attack, no defence (main-matrix wordings)
  - the 20 weak essays under every direct/fake_note wording, no defence (wording follow-up)
Each group is saved as its own .npz, so an interrupted run resumes by group.

analyse: a logistic-regression probe per layer, weak vs strong, 5-fold CV on the probe set, with
a shuffled-label control and a length-only baseline. The best layer's probe (trained on all 200
clean essays) is applied to attacked weak essays and compared with the score the model writes
(the fp16 argmax from Part A, same weights). Writes results/whitebox/probe_*.csv, fig4 and fig5.

Control: the injected text shifts *every* essay's hidden state, strong ones included (at some
layers the probe calls attacked strong essays "weak"), so "probe says weak while the output says 6"
is confounded. The cleaner test is ranking: within one attack condition, does the probe still
separate the main set's weak from strong essays (AUROC), compared with the model's own expected
score? That is insensitive to a shift shared by all essays.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "whitebox"
OUT = ROOT / "results" / "whitebox"
SEED = 0


def groups() -> list[tuple[str, list[dict], str, int | None]]:
    """(name, essays, attack, forced variant), all without a defence."""
    from assessor_eval.attacks import ATTACKS
    from assessor_eval.prompts import ASAP
    from assessor_eval.task import load_split
    import json

    probe = [json.loads(line) for line in (ASAP / "probe_set.jsonl").read_text().splitlines()]
    main = load_split("main")
    weak = [e for e in main if e["tier"] == "weak"]
    out = [("probe_clean", probe, "none", None)]
    out += [(f"main_{a}", main, a, None) for a in ATTACKS]
    out += [(f"var_{a}_v{v}", weak, a, v) for a in ("direct", "fake_note") for v in range(len(ATTACKS[a]))]
    return out


def extract() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from assessor_eval.task import build_sample
    from score_distribution import MODEL, prompt_ids, score_tokens

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    prefill, _ = score_tokens(tok)
    CACHE.mkdir(parents=True, exist_ok=True)

    cache, sys_len = None, 0
    for name, essays, attack, variant in groups():
        path = CACHE / f"{name}.npz"
        if path.exists():
            print(f"{name}: done already", flush=True)
            continue
        states, meta = [], []
        for e in essays:
            sample = build_sample(e, attack, "none", "end", variant)
            sys_ids, ids = prompt_ids(tok, sample, prefill)
            with torch.no_grad():
                if cache is None:  # system prompt is the same for every group (no defence)
                    cache = model(input_ids=torch.tensor([sys_ids], device=device), use_cache=True, logits_to_keep=1).past_key_values
                    sys_len = len(sys_ids)
                assert len(sys_ids) == sys_len
                out = model(input_ids=torch.tensor([ids[sys_len:]], device=device), past_key_values=cache,
                            use_cache=True, output_hidden_states=True, logits_to_keep=1)
            cache.crop(-(cache.get_seq_length() - sys_len))
            # hidden_states: embeddings + one per layer; keep the last position of each
            states.append(torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy())
            meta.append((e["essay_id"], e["tier"], e["true_score"], e["words"]))
            if device == "mps":
                torch.mps.empty_cache()
        ids_, tiers, trues, words = zip(*meta)
        np.savez(path, states=np.stack(states).astype(np.float16), essay_id=ids_, tier=tiers, true=trues, words=words)
        print(f"{name}: {len(states)} essays, states {np.stack(states).shape}", flush=True)


def load_group(name: str) -> dict:
    z = np.load(CACHE / f"{name}.npz")
    return {k: z[k] for k in z.files} | {"states": z["states"].astype(np.float32)}


def make_probe(C: float = 0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def analyse() -> None:
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from analyse import style
    from assessor_eval.attacks import ATTACKS, pick_variant

    probe = load_group("probe_clean")
    X_all, y = probe["states"], (probe["tier"] == "weak").astype(int)  # 1 = weak
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)
    y_shuf = rng.permutation(y)
    rows = []
    for layer in range(X_all.shape[1]):
        X = X_all[:, layer]
        rows.append({"layer": layer, "cv_acc": cross_val_score(make_probe(), X, y, cv=cv).mean(),
                     "shuffled_acc": cross_val_score(make_probe(), X, y_shuf, cv=cv).mean()})
    layers = pd.DataFrame(rows)
    length_acc = cross_val_score(make_probe(), probe["words"].reshape(-1, 1).astype(float), y, cv=cv).mean()
    best = int(layers.loc[layers.cv_acc.idxmax(), "layer"])
    print(layers.round(3).to_string(index=False))
    print(f"\nlength-only baseline: {length_acc:.3f}; best layer {best} ({layers.cv_acc.max():.3f})")

    # Fit on all 200 clean probe-set essays at the chosen layer(s), then apply to the main set
    dist = pd.read_csv(OUT / "score_dist.csv")
    hf = dist[dist.defence == "none"].set_index(["set", "attack", "variant", "essay_id"]).argmax
    chosen = sorted({best, *layers.nlargest(3, "cv_acc").layer.astype(int)})[:3]
    out = []
    for layer in chosen:
        clf = make_probe().fit(X_all[:, layer], y)
        for name, _, attack, variant in groups()[1:]:
            g = load_group(name)
            p_weak = clf.predict_proba(g["states"][:, layer])[:, 1]
            set_ = "main" if name.startswith("main") else "variants"
            for i, eid in enumerate(g["essay_id"]):
                v = variant if variant is not None else ATTACKS[attack].index(pick_variant(attack, eid))
                out.append({"layer": layer, "set": set_, "attack": attack, "variant": v, "essay_id": eid,
                            "tier": g["tier"][i], "true": int(g["true"][i]), "p_weak": p_weak[i],
                            "output": int(hf[(set_, attack, v, eid)])})
    per = pd.DataFrame(out)

    # Transfer check: the clean main set (shorter essays than the probe set) by tier
    transfer = (per[(per.set == "main") & (per.attack == "none")].groupby(["layer", "tier"])
                .p_weak.agg(probe_says_weak=lambda x: (x > 0.5).mean(), mean_p_weak="mean").reset_index())
    # Key comparison: attacked weak essays, probe says weak vs output still scores <= 2 (and >= 4)
    w = per[per.tier == "weak"]
    key = (w.groupby(["layer", "set", "attack", "variant"])
           .apply(lambda g: pd.Series({"n": len(g), "probe_says_weak": (g.p_weak > 0.5).mean(),
                                       "mean_p_weak": g.p_weak.mean(), "output_le2": (g.output <= 2).mean(),
                                       "output_ge4": (g.output >= 4).mean(),
                                       "knows_better": ((g.p_weak > 0.5) & (g.output >= 4)).sum() / max((g.output >= 4).sum(), 1)}),
                  include_groups=False).reset_index())

    # Ranking test, every layer: AUROC(weak vs strong) of the probe within each attack condition,
    # against the same AUROC for the model's expected score (Part A, same weights, no defence)
    from sklearn.metrics import roc_auc_score
    from assessor_eval.attacks import ATTACKS as ATTACK_NAMES
    surv = []
    main_groups = {a: load_group(f"main_{a}") for a in ATTACK_NAMES}
    for layer in range(1, X_all.shape[1]):
        clf = make_probe().fit(X_all[:, layer], y)
        for a, g in main_groups.items():
            keep = g["tier"] != "middling"
            surv.append({"layer": layer, "attack": a,
                         "auroc": roc_auc_score(g["tier"][keep] == "weak", clf.predict_proba(g["states"][keep, layer])[:, 1])})
    surv = pd.DataFrame(surv)
    d = dist[(dist.set == "main") & (dist.defence == "none") & (dist.tier != "middling")]
    out_auroc = {a: roc_auc_score(g.tier == "weak", -g.expected) for a, g in d.groupby("attack")}
    surv["output_auroc"] = surv.attack.map(out_auroc)
    surv.round(3).to_csv(OUT / "probe_ranking.csv", index=False)

    # Paired, tier-stratified bootstrap of (probe AUROC - output AUROC) at the CV-best layer and two
    # mid layers. Only the CV-best layer was chosen in advance; the mid layers are exploratory.
    rng, ci = np.random.default_rng(SEED), []
    for a in ("direct", "fake_note"):
        g = main_groups[a]
        keep = g["tier"] != "middling"
        lab, ids = g["tier"][keep] == "weak", g["essay_id"][keep]
        score_out = -d[d.attack == a].set_index("essay_id").loc[ids].expected.to_numpy()
        w, st = np.where(lab)[0], np.where(~lab)[0]
        boots = [np.r_[rng.choice(w, len(w)), rng.choice(st, len(st))] for _ in range(2000)]
        for layer in sorted({best, 15, 17}):
            p = make_probe().fit(X_all[:, layer], y).predict_proba(g["states"][keep, layer])[:, 1]
            diffs = [roc_auc_score(lab[i], p[i]) - roc_auc_score(lab[i], score_out[i]) for i in boots]
            ci.append({"attack": a, "layer": layer, "chosen_in_advance": layer == best,
                       "probe_auroc": roc_auc_score(lab, p), "output_auroc": roc_auc_score(lab, score_out),
                       "diff": roc_auc_score(lab, p) - roc_auc_score(lab, score_out),
                       "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5)})
    ci = pd.DataFrame(ci)
    ci.round(3).to_csv(OUT / "probe_ranking_ci.csv", index=False)
    print("\nProbe minus output ranking AUROC, 95% CI\n", ci.round(2).to_string(index=False))
    print("\nRanking AUROC, weak vs strong (main set), probe at selected layers vs model output\n",
          surv[surv.layer.isin([best, 10, 15, 17, 30, 36])].pivot(index="layer", columns="attack", values="auroc").round(2).to_string(),
          "\noutput:", {a: round(v, 2) for a, v in out_auroc.items()})

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    shades = {"none": "#1f1f1d", "direct": "#1baf7a", "fake_note": "#2a78d6", "keyword": "#8a8a84"}
    for a, g in surv.groupby("attack"):
        ax.plot(g.layer, g.auroc, color=shades[a], lw=2 if a != "keyword" else 1.5, label=f"probe, {'clean' if a == 'none' else a}")
        ax.axhline(out_auroc[a], color=shades[a], lw=1, ls="--")
    ax.axhline(0.5, color="#c9c9c4", lw=1, zorder=0)
    ax.text(36.3, 0.5, "chance", fontsize=8, color="#5c5c57", va="center")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC, weak vs strong essays")
    ax.set_ylim(0.3, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    ax.set_title("Does quality information survive the attack? Probe (solid) vs model's own score (dashed)",
                 fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_probe_ranking.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)

    layers.assign(length_only_acc=length_acc).round(3).to_csv(OUT / "probe_layers.csv", index=False)
    transfer.round(3).to_csv(OUT / "probe_transfer.csv", index=False)
    key.round(3).to_csv(OUT / "probe_attacked.csv", index=False)
    per.round(4).to_csv(OUT / "probe_per_essay.csv", index=False)

    pd.set_option("display.width", 180)
    print("\nTransfer to the clean main set\n", transfer.round(2).to_string(index=False))
    print("\nAttacked weak essays: probe vs output\n", key.round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(layers.layer, layers.cv_acc, color="#1baf7a", lw=2, label="probe (weak vs strong)")
    ax.plot(layers.layer, layers.shuffled_acc, color="#8a8a84", lw=1.5, ls="--", label="shuffled-label control")
    ax.axhline(length_acc, color="#5c5c57", lw=1, ls=":", label=f"length only ({length_acc:.2f})")
    ax.axvline(best, color="#c9c9c4", lw=1, zorder=0)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Linear probe on Qwen3-4B's last prompt token, 200 clean essays", fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_probe_layers.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT / "whitebox"), str(ROOT / "scripts")]
    {"extract": extract, "analyse": analyse}[sys.argv[1]]()
