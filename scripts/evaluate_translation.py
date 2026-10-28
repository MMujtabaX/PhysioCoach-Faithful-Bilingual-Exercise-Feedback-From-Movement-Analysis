"""Evaluate English -> Urdu translation of coaching cues.

1. Fill ``ur_reference`` in data/translation_eval.csv with your own Urdu
   translations (aim for how a trainer would actually say it).
2. Run:  python scripts/evaluate_translation.py --csv data/translation_eval.csv

Reports chrF against your references (only rows that have one), the share of
translations that keep every number and side, and the template-fallback rate.
Writes results/translation_eval.csv with the machine translation next to each
reference plus empty columns for native-speaker ratings (adequacy, fluency 1-5).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from physiocoach.translate import NLLBTranslator, check_preservation, load_glossary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/translation_eval.csv")
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--glossary", default="glossary_ur.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    import sacrebleu

    tr = NLLBTranslator(args.model, cache_path=None, glossary=load_glossary(args.glossary))
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    out_rows, hyps, refs, kept, times = [], [], [], 0, []
    for i, r in enumerate(rows, 1):
        t0 = time.perf_counter()
        mt, _ = tr.translate(r["en"])
        times.append((time.perf_counter() - t0) * 1000)
        pres = check_preservation(r["en"], mt)
        kept += pres.ok
        if r.get("ur_reference", "").strip():
            hyps.append(mt)
            refs.append(r["ur_reference"].strip())
        out_rows.append({**r, "machine_translation": mt, "preserved": pres.ok,
                         "missing_numbers": pres.missing_numbers, "side_ok": pres.side_ok,
                         "adequacy_1_5": "", "fluency_1_5": "", "comment": ""})
        if i % 20 == 0:
            print(f"  translated {i}/{len(rows)}", flush=True)

    summary = {
        "n": len(rows),
        "preserved_pct": round(100 * kept / max(1, len(rows)), 1),
        "fallback_pct": round(100 * (1 - kept / max(1, len(rows))), 1),
        "latency_ms_mean": round(sum(times) / max(1, len(times)), 1),
        "chrf_vs_reference": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2) if refs else None,
        "n_with_reference": len(refs),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "translation_eval.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    (out / "translation_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not refs:
        print("No ur_reference filled in yet: chrF is skipped. Add references and re-run.")


if __name__ == "__main__":
    main()
