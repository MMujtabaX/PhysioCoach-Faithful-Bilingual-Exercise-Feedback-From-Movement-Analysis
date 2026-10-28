"""Generate spoken cues for a listening test and measure TTS latency.

    python scripts/speech_eval.py --n 20

For N random reps it produces the final cue in English and Urdu (through the
full pipeline), synthesizes audio, and records cold (first) and warm (cached)
latency. Listeners then rate each clip in results/speech_eval/sheet.csv
(intelligibility 1-5, naturalness 1-5).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import statistics
import time
from pathlib import Path

from physiocoach.factory import Registry
from physiocoach.records import random_rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--backend", default="template")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="results/speech_eval")
    args = ap.parse_args()

    reg = Registry()
    if reg.speaker is None:
        raise SystemExit(f"TTS not available: {reg.status()['tts']['error']}")
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows, cold, warm = [], {"en": [], "ur": []}, {"en": [], "ur": []}

    for i in range(args.n):
        rec = random_rep(rng, p_fault=0.5, cfg=reg.cfg)
        for lang in ("en", "ur"):
            fb = reg.pipeline(args.backend, lang == "ur", False).rep(rec, lang)
            t0 = time.perf_counter()
            path, cached = reg.speaker.speak(fb.text, lang)
            first = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            reg.speaker.speak(fb.text, lang)
            second = (time.perf_counter() - t0) * 1000
            if not cached:
                cold[lang].append(first)
            warm[lang].append(second)
            clip = out / "clips" / f"{i:03d}_{lang}.wav"
            shutil.copy(path, clip)
            rows.append({"clip": clip.name, "lang": lang, "text": fb.text,
                         "intelligibility_1_5": "", "naturalness_1_5": "", "comment": ""})
        print(f"  {i + 1}/{args.n}", flush=True)

    rng.shuffle(rows)  # blind order for listeners
    with open(out / "sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {lang: {
        "cold_ms_mean": round(statistics.mean(cold[lang]), 1) if cold[lang] else None,
        "warm_ms_mean": round(statistics.mean(warm[lang]), 2),
    } for lang in ("en", "ur")}
    (out / "latency.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Clips and rating sheet in {out.resolve()}")


if __name__ == "__main__":
    main()
