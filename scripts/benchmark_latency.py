"""Measure cue latency for one or more Flan-T5 model folders (PyTorch or ONNX).

    python scripts/benchmark_latency.py --models models/flan-t5-coach-para-w5 models/flan-t5-coach-para-w5-onnx

For each model it times beam search (4 beams), greedy decoding and guided sampling on the same random rep
records (after a short warm-up) and reports mean / median / p95 in milliseconds,
plus the checker pass rate, so a speed-up never silently costs faithfulness.
Results are also written to results/latency.md.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

from physiocoach import check_rep, load_config
from physiocoach.generators import GuidedT5Backend, T5Backend
from physiocoach.records import random_rep


def bench(gen, records, cfg):
    times, ok = [], 0
    for r in records:
        t0 = time.perf_counter()
        text = gen.rep(r)
        times.append((time.perf_counter() - t0) * 1000)
        ok += check_rep(text, r, cfg).faithful
    times.sort()
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p95": times[int(0.95 * (len(times) - 1))],
        "faithful": 100 * ok / len(records),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=40, help="rep records to time")
    ap.add_argument("--samples", type=int, default=6, help="candidates for guided sampling")
    ap.add_argument("--out", default="results/latency.md")
    args = ap.parse_args()

    cfg = load_config()
    rng = random.Random(11)
    records = [random_rep(rng, p_fault=0.4, cfg=cfg) for _ in range(args.n)]
    rows = []
    for m in args.models:
        kind = "ONNX" if any(Path(m).glob("*encoder*.onnx")) else "PyTorch"
        for label, gen in (("beam x4", T5Backend(m, num_beams=4, cfg=cfg)),
                           ("greedy", T5Backend(m, num_beams=1, cfg=cfg)),
                           (f"guided x{args.samples}", GuidedT5Backend(m, n_samples=args.samples, seed=0, cfg=cfg))):
            for r in records[:3]:  # warm-up
                gen.rep(r)
            s = bench(gen, records, cfg)
            rows.append((Path(m).name, kind, label, s))
            print(f"{Path(m).name:32s} {kind:7s} {label:10s} mean {s['mean']:7.1f} ms  median {s['median']:7.1f}"
                  f"  p95 {s['p95']:7.1f}  faithful {s['faithful']:5.1f}%", flush=True)

    lines = ["| model | runtime | decoding | mean ms | median ms | p95 ms | faithful % |", "|---|---|---|---|---|---|---|"]
    for name, kind, label, s in rows:
        lines.append(f"| {name} | {kind} | {label} | {s['mean']:.0f} | {s['median']:.0f} | {s['p95']:.0f} | {s['faithful']:.1f} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\n\nWrote {out}")


if __name__ == "__main__":
    main()
