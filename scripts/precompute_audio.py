"""Pre-record every live coaching sentence so spoken cues play instantly.

    python scripts/precompute_audio.py                 # English + Urdu
    python scripts/precompute_audio.py --langs ur      # Urdu only

Live rep cues are assembled from a fixed set of sentences (one per correction
or praise, with every side and every degree value from 1 to --max-deficit).
This script synthesizes each one into cache/audio/. Run it once after
installing the ML extras; it can be stopped and resumed (finished sentences
are skipped). Afterwards a live spoken cue takes milliseconds instead of seconds.
Re-run it whenever you change cue wording in config/exercises.yaml.
"""

from __future__ import annotations

import argparse
import time

from physiocoach.factory import Registry
from physiocoach.templates import all_rep_sentences


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="en,ur")
    ap.add_argument("--max-deficit", type=int, default=45)
    args = ap.parse_args()

    reg = Registry()
    if reg.speaker is None:
        raise SystemExit(f"TTS not available: {reg.status()['tts']['error']}")

    for lang in args.langs.split(","):
        sentences = all_rep_sentences(reg.cfg, lang, args.max_deficit)
        todo = [s for s in sentences if not reg.speaker.cache_path(s, lang, kind="sent").exists()]
        print(f"[{lang}] {len(sentences)} sentences, {len(sentences) - len(todo)} already cached, "
              f"{len(todo)} to record")
        t0 = time.perf_counter()
        for i, s in enumerate(todo, 1):
            reg.speaker.speak_sentence(s, lang)
            if i % 25 == 0 or i == len(todo):
                rate = (time.perf_counter() - t0) / i
                print(f"  {i}/{len(todo)}  ({rate:.2f} s each, ~{rate * (len(todo) - i) / 60:.0f} min left)",
                      flush=True)
    print("Done. Live spoken cues will now come from the cache.")


if __name__ == "__main__":
    main()
