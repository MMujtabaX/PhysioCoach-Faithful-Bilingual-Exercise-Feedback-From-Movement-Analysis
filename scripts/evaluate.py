"""Compare generators on the test set: similarity, faithfulness, variety, speed.

    python scripts/evaluate.py --test data/test.jsonl --backends template,t5,llm
    python scripts/evaluate.py --test data/test_verified.jsonl --backends template,t5 --bertscore

Outputs in --out (default ``results``):
  metrics.json, metrics.md          one row per backend x task
  predictions_<backend>.csv         every input, output, reference, check result

Notes for the report
* Faithfulness is reported *before* the checker's fallback (``raw``). After the
  fallback, every output is faithful by construction; the difference between
  the two is the ablation result.
* References built from templates favour the template baseline on BLEU/ROUGE.
  Use the hand-verified set (test_verified.jsonl) for the headline numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from physiocoach import RepRecord, SessionRecord, check_rep, check_session, load_config
from physiocoach.facts import rep_facts, session_facts
from physiocoach.factory import Registry


# --------------------------------------------------------------- metrics
def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, refs: list[str]) -> float:
    p = pred.lower().split()
    best = 0.0
    for ref in refs:
        r = ref.lower().split()
        if not p or not r:
            continue
        l = _lcs(p, r)
        if l == 0:
            continue
        prec, rec = l / len(p), l / len(r)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def distinct_n(texts: list[str], n: int) -> float:
    grams, total = set(), 0
    for t in texts:
        toks = t.lower().split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i : i + n]))
            total += 1
    return len(grams) / total if total else 0.0


def corpus_scores(preds: list[str], refs: list[list[str]], bertscore: bool) -> dict:
    import sacrebleu

    k = max(len(r) for r in refs)
    streams = [[r[i] if i < len(r) else r[0] for r in refs] for i in range(k)]
    out = {
        "bleu": round(sacrebleu.corpus_bleu(preds, streams).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(preds, streams).score, 2),
        "rouge_l": round(100 * statistics.mean(rouge_l(p, r) for p, r in zip(preds, refs)), 2),
    }
    if bertscore:
        from bert_score import score

        best = []
        for i in range(k):
            _, _, f1 = score(preds, streams[i], lang="en", verbose=False)
            best.append(f1.tolist())
        out["bertscore_f1"] = round(100 * statistics.mean(max(col) for col in zip(*best)), 2)
    return out


# ------------------------------------------------------------------ main
def load_test(path: Path, tasks: set[str], limit: int | None):
    grouped: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["task"] not in tasks:
                continue
            g = grouped.setdefault(row["id"], {**row, "refs": []})
            g["refs"].append(row["target"])
    items = list(grouped.values())
    if limit:
        by_task = defaultdict(list)
        for it in items:
            by_task[it["task"]].append(it)
        items = [it for t in by_task.values() for it in t[:limit]]
    return items


def evaluate_backend(name, gen, items, cfg, bertscore):
    rows, by_task = [], defaultdict(list)
    for it in items:
        obj = RepRecord(**it["record"]) if it["task"] == "rep" else SessionRecord(**it["record"])
        t0 = time.perf_counter()
        pred = gen.rep(obj) if it["task"] == "rep" else gen.session(obj)
        ms = (time.perf_counter() - t0) * 1000
        res = check_rep(pred, obj, cfg) if it["task"] == "rep" else check_session(pred, obj, cfg)
        dec = dict(getattr(gen, "last_stats", {}) or {})
        if it["task"] == "rep":
            f = rep_facts(obj, cfg)
            truth, required = set(f.present), set(f.required)
        else:
            f = session_facts(obj, cfg)
            truth, required = set(f.occurred), set(f.notable)
        row = {"id": it["id"], "task": it["task"], "input": it["input"], "prediction": pred,
               "references": " || ".join(it["refs"]), "faithful": res.faithful,
               "issues": "; ".join(res.issues), "latency_ms": round(ms, 1),
               "_mentioned": set(res.mentioned), "_truth": truth, "_required": required,
               "_hall": bool(res.hallucinated or res.wrong_praise), "_num": bool(res.unexpected_numbers),
               "_side": res.side_error, "_refs": it["refs"], "_dec": dec}
        rows.append(row)
        by_task[it["task"]].append(row)

    metrics = {}
    for task, rs in by_task.items():
        mentioned = sum(len(r["_mentioned"]) for r in rs)
        correct = sum(len(r["_mentioned"] & r["_truth"]) for r in rs)
        required = sum(len(r["_required"]) for r in rs)
        covered = sum(len(r["_mentioned"] & r["_required"]) for r in rs)
        lat = sorted(r["latency_ms"] for r in rs)
        m = {
            "n": len(rs),
            **corpus_scores([r["prediction"] for r in rs], [r["_refs"] for r in rs], bertscore),
            "faithful_raw_pct": round(100 * statistics.mean(r["faithful"] for r in rs), 1),
            "fault_precision": round(100 * correct / mentioned, 1) if mentioned else 100.0,
            "fault_recall": round(100 * covered / required, 1) if required else 100.0,
            "hallucination_pct": round(100 * statistics.mean(r["_hall"] for r in rs), 1),
            "number_error_pct": round(100 * statistics.mean(r["_num"] for r in rs), 1),
            "side_error_pct": round(100 * statistics.mean(r["_side"] for r in rs), 1),
            "fallback_pct": round(100 * statistics.mean(not r["faithful"] for r in rs), 1),
            "faithful_after_check_pct": 100.0,
            "distinct_1": round(distinct_n([r["prediction"] for r in rs], 1), 3),
            "distinct_2": round(distinct_n([r["prediction"] for r in rs], 2), 3),
            "latency_ms_mean": round(statistics.mean(lat), 1),
            "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1),
        }
        decs = [r["_dec"] for r in rs if r["_dec"].get("candidates")]
        if decs:  # checker-guided decoding statistics
            m["guided_distinct_per_record"] = round(statistics.mean(d["distinct"] for d in decs), 2)
            m["guided_faithful_per_record"] = round(statistics.mean(d["faithful"] for d in decs), 2)
            m["guided_rejected_pct"] = round(100 * (1 - sum(d["faithful"] for d in decs)
                                                    / max(1, sum(d["distinct"] for d in decs))), 1)
        metrics[task] = m
    return metrics, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--backends", default="template,t5")
    ap.add_argument("--tasks", default="rep,session")
    ap.add_argument("--limit", type=int, default=None, help="max records per task")
    ap.add_argument("--bertscore", action="store_true", help="also compute BERTScore (downloads roberta-large)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed for t5-guided")
    args = ap.parse_args()

    cfg = load_config()
    reg = Registry(cfg=cfg)
    items = load_test(Path(args.test), set(args.tasks.split(",")), args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(items)} test records from {args.test}")

    all_metrics = {}
    for name in args.backends.split(","):
        try:
            gen = reg.generator(name)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        if hasattr(gen, "rng"):
            gen.rng.seed(args.seed)
        print(f"  running {name}...", flush=True)
        metrics, rows = evaluate_backend(name, gen, items, cfg, args.bertscore)
        all_metrics[name] = metrics
        with open(out / f"predictions_{name}.csv", "w", newline="", encoding="utf-8") as fh:
            cols = ["id", "task", "input", "prediction", "references", "faithful", "issues", "latency_ms"]
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    cols = ["n", "bleu", "chrf", "rouge_l", "bertscore_f1", "faithful_raw_pct", "fault_precision",
            "fault_recall", "hallucination_pct", "number_error_pct", "distinct_2", "latency_ms_mean",
            "guided_rejected_pct"]
    lines = ["| backend | task | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for name, by_task in all_metrics.items():
        for task, m in by_task.items():
            lines.append(f"| {name} | {task} | " + " | ".join(str(m.get(c, "")) for c in cols) + " |")
    table = "\n".join(lines)
    (out / "metrics.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nWrote results to {out.resolve()}")


if __name__ == "__main__":
    main()
