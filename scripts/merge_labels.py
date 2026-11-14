"""Merge my blind labels (data/synthetic/labels.csv) into data/synthetic/applications.jsonl as true_score.

    uv run python scripts/merge_labels.py
"""

import csv
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "synthetic"


def main() -> None:
    apps_path = DATA / "applications.jsonl"
    apps = [json.loads(line) for line in apps_path.read_text().splitlines() if line.strip()]
    labels = {r["id"]: r["true_score"].strip() for r in csv.DictReader((DATA / "labels.csv").open())}

    errors = []
    for app in apps:
        raw = labels.get(app["id"], "")
        if raw not in {"1", "2", "3", "4", "5"}:
            errors.append(f"{app['id']}: score {raw!r} is missing or not 1-5")
    extra = set(labels) - {a["id"] for a in apps}
    errors += [f"{id_}: in labels.csv but not in applications.jsonl" for id_ in sorted(extra)]
    if errors:
        raise SystemExit("Not merged:\n  " + "\n  ".join(errors))

    for app in apps:
        app["true_score"] = int(labels[app["id"]])
    apps_path.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in apps))

    tiers = {"weak (1-2)": 0, "middling (3)": 0, "strong (4-5)": 0}
    for app in apps:
        s = app["true_score"]
        tiers["weak (1-2)" if s <= 2 else "middling (3)" if s == 3 else "strong (4-5)"] += 1
    print(f"merged {len(apps)} labels: " + ", ".join(f"{k} n={v}" for k, v in tiers.items()))


if __name__ == "__main__":
    main()
