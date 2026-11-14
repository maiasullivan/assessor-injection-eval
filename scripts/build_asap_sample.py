"""Build the essay samples from the raw ASAP 2.0 CSV (not committed; see README).

    uv run python scripts/build_asap_sample.py

Writes (all from the "Exploring Venus" prompt only):
  data/asap/task.json         assignment text + source article (identical for every essay)
  data/asap/essays.jsonl      main set: 60 essays stratified by human score
  data/asap/dev.jsonl         dev set: 20 more essays, the only ones prompt changes may be tuned on
  data/asap/probe_set.jsonl   Part B: 100 weak (1-2) + 100 strong (4-6), length-matched, disjoint from main set

Only `essay_id`, the essay text and the human score are kept. Demographic columns are dropped.
The score and id are stored for scoring/analysis only and must never be put in a model input.
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


def stratified(df: pd.DataFrame, per_score: dict[int, int]) -> pd.DataFrame:
    parts = []
    for score, n in per_score.items():
        s = df[df.score == score]
        lo, hi = s.words.quantile([0.05, 0.95])
        parts.append(s[s.words.between(lo, hi)].sample(n, random_state=SEED))
    return pd.concat(parts).sample(frac=1, random_state=SEED)


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
    rest = df[~df.essay_id.isin(used) & df.words.between(*PROBE_BAND) & (df.score != 3)].copy()
    rest["bin"] = rest.words // PROBE_BIN
    rest["cls"] = (rest.score >= 4).map({True: "strong", False: "weak"})
    counts = rest.groupby(["bin", "cls"]).size().unstack(fill_value=0)
    per_bin = counts.min(axis=1)
    # scale bin quotas down to PROBE_PER_CLASS in total
    quota = (per_bin * PROBE_PER_CLASS / per_bin.sum()).round().astype(int)
    quota.iloc[quota.argmax()] += PROBE_PER_CLASS - quota.sum()
    probe_parts = [rest[(rest.bin == b) & (rest.cls == c)].sample(q, random_state=SEED)
                   for b, q in quota.items() if q for c in ("weak", "strong")]
    probe = pd.concat(probe_parts).sample(frac=1, random_state=SEED)
    write_jsonl(OUT / "probe_set.jsonl", [row(r) for r in probe.itertuples()])

    print(f"dev: {len(dev)} essays, by score {dev.score.value_counts().sort_index().to_dict()}")
    print(f"main: {len(main)} essays, by score {main.score.value_counts().sort_index().to_dict()}")
    print(f"main words by tier: {main.groupby(main.score.map(tier)).words.median().to_dict()}")
    print(f"probe: {len(probe)} essays, by class {probe.cls.value_counts().to_dict()}, "
          f"median words {probe.groupby('cls').words.median().to_dict()}")


if __name__ == "__main__":
    main()
