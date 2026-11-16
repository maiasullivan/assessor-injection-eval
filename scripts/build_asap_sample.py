"""Build the essay sets from the raw ASAP 2.0 csv ("Exploring Venus" prompt only).

    uv run python scripts/build_asap_sample.py

Writes data/asap/: task.json, essays.jsonl (main, 60), dev.jsonl (20), probe_set.jsonl (200,
length-matched), confirm.jsonl and confirm_lm.jsonl (120 each). All sets are disjoint.
Demographic columns are dropped. See docs/methods.md for how each set is drawn.
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "ASAP2_train_sourcetexts.csv"
OUT = ROOT / "data" / "asap"
PROMPT = "Exploring Venus"
SEED = 0

# main set: number of essays per human score (weak 1-2, middling 3, strong 4-6)
MAIN_PER_SCORE = {1: 10, 2: 10, 3: 20, 4: 8, 5: 7, 6: 5}
DEV_PER_SCORE = {1: 3, 2: 4, 3: 6, 4: 4, 5: 2, 6: 1}
# probe set: length band where weak and strong essays overlap, and bin width for matching
PROBE_BAND, PROBE_BIN, PROBE_PER_CLASS = (280, 480), 20, 100
# confirmation set for Part B: weak and strong only, roughly in proportion to the prompt's score counts
CONFIRM_PER_SCORE, CONFIRM_SEED = {1: 18, 2: 42, 4: 45, 5: 12, 6: 3}, 1
CONFIRM_LM_BAND, CONFIRM_LM_PER_CLASS, CONFIRM_LM_SEED = (250, 500), 60, 2


def normalise(text: str) -> str:
    """Strip export artefacts (trailing/multiple spaces) without touching wording or spelling."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tier(score: int) -> str:
    return "weak" if score <= 2 else "middling" if score == 3 else "strong"


def row(r) -> dict:
    return {"essay_id": r.essay_id, "text": r.text, "true_score": int(r.score), "tier": tier(r.score), "words": int(r.words)}


def stratified(df: pd.DataFrame, per_score: dict[int, int], seed: int = SEED) -> pd.DataFrame:
    parts = []
    for score, n in per_score.items():
        s = df[df.score == score]
        lo, hi = s.words.quantile([0.05, 0.95])
        parts.append(s[s.words.between(lo, hi)].sample(n, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed)


def length_matched(df: pd.DataFrame, band: tuple[int, int], per_class: int, seed: int) -> pd.DataFrame:
    """Equal numbers of weak and strong essays per 20-word length bin, so length can't separate them."""
    rest = df[df.words.between(*band) & (df.score != 3)].copy()
    rest["bin"] = rest.words // PROBE_BIN
    rest["cls"] = (rest.score >= 4).map({True: "strong", False: "weak"})
    counts = rest.groupby(["bin", "cls"]).size().unstack(fill_value=0)
    per_bin = counts.min(axis=1)
    # scale bin quotas down to per_class in total
    quota = (per_bin * per_class / per_bin.sum()).round().astype(int)
    quota.iloc[quota.argmax()] += per_class - quota.sum()
    parts = [rest[(rest.bin == b) & (rest.cls == c)].sample(q, random_state=seed)
             for b, q in quota.items() if q for c in ("weak", "strong")]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    df = pd.read_csv(RAW)
    df = df[df.prompt_name == PROMPT].copy()
    df["text"] = df.full_text.map(normalise)
    df["words"] = df.text.str.split().str.len()

    OUT.mkdir(parents=True, exist_ok=True)
    sources = [s.strip() for s in df.iloc[0][[c for c in df.columns if c.startswith("source_text")]].dropna()]
    task = {"prompt_name": PROMPT, "assignment": df.assignment.iloc[0].strip(), "source_texts": sources}
    (OUT / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n")

    # Main and dev sets: within each score, drop length outliers (outside the 5th-95th percentile) then sample.
    main = stratified(df, MAIN_PER_SCORE)
    dev = stratified(df[~df.essay_id.isin(main.essay_id)], DEV_PER_SCORE)
    write_jsonl(OUT / "essays.jsonl", [row(r) for r in main.itertuples()])
    write_jsonl(OUT / "dev.jsonl", [row(r) for r in dev.itertuples()])

    # Probe set: equal numbers of weak and strong essays per length bin, so length can't separate them.
    used = set(main.essay_id) | set(dev.essay_id)
    probe = length_matched(df[~df.essay_id.isin(used)], PROBE_BAND, PROBE_PER_CLASS, SEED)
    write_jsonl(OUT / "probe_set.jsonl", [row(r) for r in probe.itertuples()])

    # Confirmation set: essays not in any earlier set, drawn like the main set (lengths not matched)
    used |= set(probe.essay_id)
    confirm = stratified(df[~df.essay_id.isin(used)], CONFIRM_PER_SCORE, seed=CONFIRM_SEED)
    write_jsonl(OUT / "confirm.jsonl", [row(r) for r in confirm.itertuples()])

    # Length-matched confirmation set: like the probe set, from essays not used anywhere else
    used |= set(confirm.essay_id)
    confirm_lm = length_matched(df[~df.essay_id.isin(used)], CONFIRM_LM_BAND, CONFIRM_LM_PER_CLASS, CONFIRM_LM_SEED)
    write_jsonl(OUT / "confirm_lm.jsonl", [row(r) for r in confirm_lm.itertuples()])

    print(f"dev: {len(dev)} essays, by score {dev.score.value_counts().sort_index().to_dict()}")
    print(f"main: {len(main)} essays, by score {main.score.value_counts().sort_index().to_dict()}")
    print(f"main words by tier: {main.groupby(main.score.map(tier)).words.median().to_dict()}")
    print(f"probe: {len(probe)} essays, by class {probe.cls.value_counts().to_dict()}, "
          f"median words {probe.groupby('cls').words.median().to_dict()}")
    print(f"confirm_lm: {len(confirm_lm)} essays, by score {confirm_lm.score.value_counts().sort_index().to_dict()}, "
          f"median words {confirm_lm.groupby('cls').words.median().to_dict()}")
    print(f"confirm: {len(confirm)} essays, by score {confirm.score.value_counts().sort_index().to_dict()}, "
          f"median words {confirm.groupby(confirm.score.map(tier)).words.median().to_dict()}")


if __name__ == "__main__":
    main()
