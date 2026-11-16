"""Build the CV sets from the Djinni dataset: Python candidates, tiered by stated experience.

    uv run --group cv python scripts/build_cv_sample.py

weak = 0-1 years, middling = 2-3, strong = 6+ (the job ad asks for 5+). CVs with an email,
phone number or link are dropped. CV text goes to data/cv/candidates*.jsonl, which is
gitignored; only ids and tiers are committed.
"""

import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data" / "cv"
SEED = 0
TIERS = {"weak": (0, 1), "middling": (2, 3), "strong": (6, 99)}
MAIN_PER_TIER = 20
DEV_PER_TIER = {"weak": 3, "middling": 4, "strong": 3}
PII = re.compile(r"[\w.+-]+@[\w-]+\.\w|\+?\d[\d \-()]{8,}\d|https?://|www\.|\.com\b|linkedin|github|t\.me|telegram", re.I)


def profile_text(r) -> str:
    """The candidate as a recruiter sees them on the platform: headline, structured fields, CV."""
    return (f"Position: {r['Position']}\nExperience: {r['Experience Years']:g} years\n"
            f"English level: {r['English Level']}\n\n{r['CV'].replace(chr(13), '').strip()}")


def tier(years: float) -> str | None:
    return next((t for t, (lo, hi) in TIERS.items() if lo <= years <= hi), None)


def main() -> None:
    df = load_dataset("lang-uk/recruitment-dataset-candidate-profiles-english")["train"].to_pandas()
    df = df[df["Primary Keyword"] == "Python"].copy()
    df["words"] = df.CV.str.split().str.len()
    df["tier"] = df["Experience Years"].map(tier)
    keep = df.tier.notna() & df.words.between(60, 400) & ~df.CV.str.contains(PII) & ~df.Position.str.contains(PII)
    df = df[keep]
    print(f"eligible Python candidates: {len(df)} by tier {df.tier.value_counts().to_dict()}")

    main_set = df.groupby("tier").sample(MAIN_PER_TIER, random_state=SEED)
    rest = df[~df.id.isin(main_set.id)]
    dev = pd.concat(rest[rest.tier == t].sample(n, random_state=SEED) for t, n in DEV_PER_TIER.items())

    for name, part in [("main", main_set), ("dev", dev)]:
        part = part.sample(frac=1, random_state=SEED)
        rows = [{"cand_id": r["id"], "text": profile_text(r), "tier": r["tier"],
                 "experience_years": float(r["Experience Years"]), "english": r["English Level"], "words": int(r["words"])}
                for _, r in part.iterrows()]
        suffix = "" if name == "main" else "_dev"
        (OUT / f"candidates{suffix}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        (OUT / f"ids{suffix}.jsonl").write_text("".join(json.dumps({k: x[k] for k in ("cand_id", "tier", "experience_years")}) + "\n" for x in rows))
        print(f"{name}: {len(rows)} by tier {part.tier.value_counts().to_dict()}, "
              f"median words {part.groupby('tier').words.median().to_dict()}")


if __name__ == "__main__":
    main()
