"""Blind human evaluation of generated feedback.

Make a rating sheet (outputs from all backends, shuffled, backend hidden):

    python scripts/human_eval.py make --predictions results/predictions_*.csv --n 50

Each rater fills clarity / actionability / naturalness (1-5) in a copy of
results/human_eval_sheet.csv. Then score all filled copies:

    python scripts/human_eval.py score results/rater_*.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

CRITERIA = ["clarity_1_5", "actionability_1_5", "naturalness_1_5"]


def make(args):
    rng = random.Random(args.seed)
    files = sorted({f for pattern in args.predictions for f in glob.glob(pattern)})
    per_backend = {}
    for f in files:
        backend = Path(f).stem.replace("predictions_", "")
        rows = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["task"] in args.tasks.split(",")]
        per_backend[backend] = {r["id"]: r for r in rows}
    common = sorted(set.intersection(*(set(v) for v in per_backend.values())))
    ids = rng.sample(common, min(args.n, len(common)))

    items = []
    for rid in ids:
        for backend, rows in per_backend.items():
            r = rows[rid]
            items.append({"input": r["input"], "feedback": r["prediction"], "_backend": backend, "_id": rid})
    rng.shuffle(items)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "human_eval_sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "input", "feedback", *CRITERIA, "comment"])
        for i, it in enumerate(items):
            w.writerow([i, it["input"], it["feedback"], "", "", "", ""])
    with open(out / "human_eval_key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "backend", "id"])
        for i, it in enumerate(items):
            w.writerow([i, it["_backend"], it["_id"]])
    print(f"{len(items)} items ({len(ids)} records x {len(per_backend)} backends) -> {out / 'human_eval_sheet.csv'}")
    print("Give raters the sheet only; keep human_eval_key.csv hidden until scoring.")


def score(args):
    key = {r["item"]: r["backend"] for r in csv.DictReader(open(args.key, encoding="utf-8"))}
    scores = defaultdict(lambda: defaultdict(list))
    raters = 0
    for f in args.sheets:
        raters += 1
        for r in csv.DictReader(open(f, encoding="utf-8")):
            for c in CRITERIA:
                if r[c].strip():
                    scores[key[r["item"]]][c].append(float(r[c]))
    result = {b: {c: round(statistics.mean(v), 2) for c, v in cs.items()} for b, cs in scores.items()}
    print(json.dumps({"raters": raters, "mean_scores": result}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--predictions", nargs="+", default=["results/predictions_*.csv"])
    m.add_argument("--n", type=int, default=50)
    m.add_argument("--tasks", default="rep")
    m.add_argument("--seed", type=int, default=3)
    m.add_argument("--out", default="results")
    m.set_defaults(func=make)
    s = sub.add_parser("score")
    s.add_argument("sheets", nargs="+")
    s.add_argument("--key", default="results/human_eval_key.csv")
    s.set_defaults(func=score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
