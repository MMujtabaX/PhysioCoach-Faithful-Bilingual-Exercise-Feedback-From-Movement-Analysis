"""Turn the reviewed CSV into the hand-verified test set.

After the team edits ``data/test_to_verify.csv`` (fix or rewrite
``verified_reference`` and put ``y`` in ``reviewed``), run:

    python scripts/import_verified.py --csv data/test_to_verify.csv --test data/test.jsonl \
        --out data/test_verified.jsonl

Rows not marked reviewed are skipped. Each verified reference is run through the
faithfulness checker; failures are reported so reviewers can double-check them
(a human may phrase something the checker's vocabulary does not know).
"""

from __future__ import annotations

import argparse
import csv
import json

from physiocoach import RepRecord, SessionRecord, check_rep, check_session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/test_to_verify.csv")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out", default="data/test_verified.jsonl")
    args = ap.parse_args()

    records = {}
    with open(args.test, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            records.setdefault(row["id"], row)

    kept, flagged = 0, []
    with open(args.csv, encoding="utf-8") as fh, open(args.out, "w", encoding="utf-8") as out:
        for row in csv.DictReader(fh):
            if row["reviewed"].strip().lower() not in {"y", "yes", "1", "true"}:
                continue
            base = records[row["id"]]
            ref = row["verified_reference"].strip()
            obj = RepRecord(**base["record"]) if base["task"] == "rep" else SessionRecord(**base["record"])
            res = check_rep(ref, obj) if base["task"] == "rep" else check_session(ref, obj)
            if not res.faithful:
                flagged.append((row["id"], ref, res.issues))
            out.write(json.dumps({**base, "target": ref, "ref_index": 0, "verified": True}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} verified references to {args.out}")
    if flagged:
        print(f"\n{len(flagged)} verified references fail the checker (review them, or extend the "
              "detection regexes in config/exercises.yaml if the wording is fine):")
        for rid, ref, issues in flagged[:20]:
            print(f"  {rid}: {ref!r} -> {issues}")


if __name__ == "__main__":
    main()
