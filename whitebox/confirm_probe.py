"""Part B confirmation tests, as pre-registered in docs/partB_probe.md (sections 4 and 9).

    uv run --group whitebox python whitebox/confirm_probe.py extract confirm      # or confirm_lm
    uv run --group whitebox python whitebox/confirm_probe.py analyse confirm

Writes results/whitebox/<set>_results.csv, <set>_per_essay.csv and fig6 (fig7 for confirm_lm).
"""

import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "whitebox"), str(ROOT / "scripts")]
from linear_probe import CACHE, OUT, SEED, load_group, make_probe  # noqa: E402

CONDITIONS = ("none", "direct", "fake_note")
MID_LAYERS, SHALLOW_LAYER = (15, 16, 17), 1
N_BOOT = 2000


def confirm_essays(name: str) -> list[dict]:
    import json
    from assessor_eval.prompts import ASAP
    return [json.loads(line) for line in (ASAP / f"{name}.jsonl").read_text().splitlines()]


def extract(name: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from assessor_eval.task import build_sample
    from score_distribution import MODEL, prompt_ids, score_tokens, summarise

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    prefill, digits = score_tokens(tok)
    CACHE.mkdir(parents=True, exist_ok=True)
    essays = confirm_essays(name)

    cache, sys_len = None, 0
    for attack in CONDITIONS:
        path = CACHE / f"{name}_{attack}.npz"
        if path.exists():
            print(f"{attack}: done already", flush=True)
            continue
        states, expected = [], []
        for e in essays:
            sample = build_sample(e, attack, "none", "end")
            sys_ids, ids = prompt_ids(tok, sample, prefill)
            with torch.no_grad():
                if cache is None:
                    cache = model(input_ids=torch.tensor([sys_ids], device=device), use_cache=True, logits_to_keep=1).past_key_values
                    sys_len = len(sys_ids)
                assert len(sys_ids) == sys_len
                out = model(input_ids=torch.tensor([ids[sys_len:]], device=device), past_key_values=cache,
                            use_cache=True, output_hidden_states=True, logits_to_keep=1)
            cache.crop(-(cache.get_seq_length() - sys_len))
            states.append(torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy())
            logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)[digits].cpu()
            expected.append(summarise(logp)["expected"])
            if device == "mps":
                torch.mps.empty_cache()
        np.savez(path, states=np.stack(states).astype(np.float16), expected=np.array(expected),
                 essay_id=[e["essay_id"] for e in essays], tier=[e["tier"] for e in essays],
                 true=[e["true_score"] for e in essays], words=[e["words"] for e in essays])
        print(f"{attack}: {len(states)} essays", flush=True)


# surface features, as listed in the pre-registration

@lru_cache(maxsize=1)
def dictionary() -> frozenset[str]:
    return frozenset(w.strip().lower() for w in Path("/usr/share/dict/words").read_text().splitlines())


def in_dict(word: str) -> bool:
    d = dictionary()
    if word in d:
        return True
    for suffix in ("s", "es", "ed", "ing", "ly"):
        if word.endswith(suffix) and word[: -len(suffix)] in d:
            return True
    return False


def surface_features(text: str) -> list[float]:
    words = text.split()
    alpha = [w.lower() for w in re.findall(r"[A-Za-z]+", text)]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    first = [w.lower().strip(".,!?;:\"'()") for w in words[:150]]
    return [
        np.log(len(words)),
        np.mean([len(w) for w in alpha]),
        len(set(first)) / len(first),
        np.mean([not in_dict(w) for w in alpha]),
        len(words) / len(sentences),
        np.mean([s[0].islower() for s in sentences]),
        100 * len(re.findall(r"(?<![A-Za-z'])i(?![A-Za-z'])", text)) / len(words),
    ]


SURFACE_NAMES = ["log_words", "mean_word_len", "ttr150", "misspelled", "sent_len", "lower_start", "lower_i_per100"]


def auroc(weak: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(weak, score)


def boot_diff(weak: np.ndarray, a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    """AUROC(a) - AUROC(b) with a paired, tier-stratified bootstrap CI."""
    w, s = np.where(weak)[0], np.where(~weak)[0]
    diffs = []
    for _ in range(N_BOOT):
        i = np.r_[rng.choice(w, len(w)), rng.choice(s, len(s))]
        diffs.append(auroc(weak[i], a[i]) - auroc(weak[i], b[i]))
    return auroc(weak, a) - auroc(weak, b), *np.percentile(diffs, [2.5, 97.5])


def oof_auroc_scores(X: np.ndarray, weak: np.ndarray, rng_seed: int) -> np.ndarray:
    """Out-of-fold p(weak) from 5-fold CV repeated 20 times (averaged over repeats)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    preds = np.zeros(len(weak))
    for rep in range(20):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rng_seed + rep).split(X, weak):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], weak[tr])
            preds[te] += clf.predict_proba(X[te])[:, 1] / 20
    return preds


def analyse(name: str) -> None:
    import matplotlib.pyplot as plt
    from figstyle import COLORS, INK, PAPER, RULE, WIDTH, save, style
    from assessor_eval.prompts import ASAP
    import json

    rng = np.random.default_rng(SEED)
    probe = load_group("probe_clean")
    y = (probe["tier"] == "weak").astype(int)
    probes = {l: make_probe().fit(probe["states"][:, l], y) for l in (*MID_LAYERS, SHALLOW_LAYER)}

    texts = {e["essay_id"]: e["text"] for e in confirm_essays(name)}
    probe_texts = {json.loads(l)["essay_id"]: json.loads(l)["text"] for l in (ASAP / "probe_set.jsonl").read_text().splitlines()}
    S_train = np.array([surface_features(probe_texts[i]) for i in probe["essay_id"]])
    surface_clf = make_probe(C=1.0).fit(S_train, y)

    rows, per_essay = [], []
    for attack in CONDITIONS:
        g = load_group(f"{name}_{attack}")
        weak = g["tier"] == "weak"
        mid = np.mean([probes[l].predict_proba(g["states"][:, l])[:, 1] for l in MID_LAYERS], axis=0)
        shallow = probes[SHALLOW_LAYER].predict_proba(g["states"][:, SHALLOW_LAYER])[:, 1]
        output = -g["expected"]  # lower expected score = more likely weak
        S = np.array([surface_features(texts[i]) for i in g["essay_id"]])
        surf_base = surface_clf.predict_proba(S)[:, 1]
        logit = np.log(np.clip(mid, 1e-6, 1 - 1e-6) / np.clip(1 - mid, 1e-6, 1))
        oof_a = oof_auroc_scores(S, weak, SEED)
        oof_b = oof_auroc_scores(np.c_[S, logit], weak, SEED)

        row = {"attack": attack, "n_weak": int(weak.sum()), "n_strong": int((~weak).sum()),
               "auroc_output": auroc(weak, output), "auroc_mid": auroc(weak, mid), "auroc_shallow": auroc(weak, shallow),
               "auroc_surface_probe_set": auroc(weak, surf_base), "auroc_surface_cv": auroc(weak, oof_a),
               "auroc_surface_plus_mid_cv": auroc(weak, oof_b), "auroc_word_count": auroc(weak, -S[:, 0]),
               "mean_expected_weak": g["expected"][weak].mean(), "pass_rate_weak": (g["expected"][weak] >= 3.5).mean()}
        for hyp, a, b in [("H1_mid_minus_output", mid, output), ("H3_mid_minus_shallow", mid, shallow),
                           ("H4_surface_plus_mid_minus_surface", oof_b, oof_a)]:
            d, lo, hi = boot_diff(weak, a, b, rng)
            row |= {hyp: d, f"{hyp}_lo": lo, f"{hyp}_hi": hi}
        rows.append(row)
        per_essay.append(pd.DataFrame({"attack": attack, "essay_id": g["essay_id"], "tier": g["tier"], "true": g["true"],
                                       "expected": g["expected"], "p_weak_mid": mid, "p_weak_shallow": shallow,
                                       **{n: S[:, k] for k, n in enumerate(SURFACE_NAMES)}}))
    res = pd.DataFrame(rows)
    res.round(3).to_csv(OUT / f"{name}_results.csv", index=False)
    pd.concat(per_essay).round(4).to_csv(OUT / f"{name}_per_essay.csv", index=False)

    pd.set_option("display.width", 200)
    print(res.T.round(3).to_string())

    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    series = [("auroc_output", "model's written score", INK), ("auroc_mid", "probe, layers 15–17", COLORS["qwen"]),
              ("auroc_shallow", "probe, layer 1", COLORS["haiku"]), ("auroc_surface_probe_set", "surface features only", "#a8a8a8")]
    x = np.arange(len(CONDITIONS))
    width = 0.2
    for k, (col, label, color) in enumerate(series):
        bars = ax.bar(x + (k - 1.5) * width, res[col], width - 0.03, color=color, label=label, edgecolor=PAPER, lw=0)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK, padding=1.5)
    ax.axhline(0.5, color=RULE, lw=0.6, ls=":", zorder=0)
    ax.set_xticks(x, ["clean", "direct", "fake note"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("AUROC, weak vs strong")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=1.0)
    style(ax)
    fig.tight_layout()
    save(fig, OUT / ("fig6_confirm.png" if name == "confirm" else f"fig7_{name}.png"))

if __name__ == "__main__":
    {"extract": extract, "analyse": analyse}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else "confirm")
