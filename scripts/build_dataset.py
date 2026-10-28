"""Build the input -> feedback dataset.

    python scripts/build_dataset.py --out data
    python scripts/build_dataset.py --out data --paraphrase --paraphrase-model qwen2.5:3b

What it does
1. Samples synthetic rep and session records (feature values agree with labels).
2. Writes several *different* template references per record (wording variants).
3. Optionally adds LLM paraphrases (Ollama) for more natural variety. Every
   paraphrase is run through the faithfulness checker and dropped if it fails.
4. Splits by record (never by sentence), 80/10/10, so no test record leaks
   into training.
5. Writes review sheets for the hand-verified test set and the translation
   evaluation.

Outputs (in --out): train.jsonl, val.jsonl, test.jsonl, stats.json,
test_to_verify.csv, translation_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from physiocoach import TemplateGenerator, check_rep, check_session, linearize_rep, linearize_session, load_config
from physiocoach.records import random_rep, random_session


def build_examples(args, cfg):
    rng = random.Random(args.seed)
    tmpl = TemplateGenerator(cfg)
    examples = []  # (record_id, task, record_obj, input, targets)

    for i in range(args.n_rep):
        rec = random_rep(rng, p_fault=rng.choice([0.12, 0.3, 0.55]), cfg=cfg)
        refs = []
        for k in range(args.refs * 3):  # sample more than needed, keep distinct ones
            t = tmpl.rep(rec, "en", rng=random.Random(f"{args.seed}-{i}-{k}"))
            if t not in refs:
                refs.append(t)
            if len(refs) == args.refs:
                break
        examples.append((f"rep-{i:05d}", "rep", rec, linearize_rep(rec, cfg), refs))

    for i in range(args.n_session):
        s = random_session(rng, cfg=cfg)
        refs = []
        for k in range(args.refs * 3):
            t = tmpl.session(s, "en", rng=random.Random(f"{args.seed}-s{i}-{k}"))
            if t not in refs:
                refs.append(t)
            if len(refs) == args.refs:
                break
        examples.append((f"session-{i:05d}", "session", s, linearize_session(s, cfg), refs))
    return examples


def paraphrase(examples, args, cfg):
    from physiocoach.generators import BackendUnavailable, OllamaBackend

    if not OllamaBackend.is_running(args.ollama_host):
        sys.exit(f"Ollama is not running at {args.ollama_host}. Start it with 'ollama serve'.")
    llm = OllamaBackend(args.paraphrase_model, args.ollama_host, temperature=0.8, cfg=cfg)
    rng = random.Random(args.seed + 1)
    kept = dropped = 0
    reps = [e for e in examples if e[1] == "rep"]
    chosen = rng.sample(reps, int(len(reps) * args.paraphrase_frac))
    for n, (_, _, rec, _, refs) in enumerate(chosen, 1):
        try:
            p = llm.paraphrase(refs[0])
        except BackendUnavailable as e:
            sys.exit(str(e))
        if p and p not in refs and check_rep(p, rec, cfg).faithful:
            refs.append(p)
            kept += 1
        else:
            dropped += 1
        if n % 50 == 0:
            print(f"  paraphrased {n}/{len(chosen)} (kept {kept}, dropped {dropped})", flush=True)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-rep", type=int, default=4000, help="rep records")
    ap.add_argument("--n-session", type=int, default=800, help="session records")
    ap.add_argument("--refs", type=int, default=3, help="template references per record")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--paraphrase", action="store_true", help="add LLM paraphrases via Ollama")
    ap.add_argument("--paraphrase-model", default="qwen2.5:3b")
    ap.add_argument("--paraphrase-frac", type=float, default=0.3)
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--verify-rep", type=int, default=200, help="rep rows in the review sheet")
    ap.add_argument("--verify-session", type=int, default=50)
    ap.add_argument("--translation-n", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.n_rep} rep and {args.n_session} session records...")
    examples = build_examples(args, cfg)

    para_stats = None
    if args.paraphrase:
        print("Paraphrasing with the LLM (filtered by the faithfulness checker)...")
        para_stats = paraphrase(examples, args, cfg)

    # sanity: every reference must be faithful
    for rid, task, obj, _, refs in examples:
        for r in refs:
            res = check_rep(r, obj, cfg) if task == "rep" else check_session(r, obj, cfg)
            assert res.faithful, f"unfaithful reference for {rid}: {r} -> {res.issues}"

    rng = random.Random(args.seed + 2)
    rng.shuffle(examples)
    n = len(examples)
    cut1, cut2 = int(0.8 * n), int(0.9 * n)
    splits = {"train": examples[:cut1], "val": examples[cut1:cut2], "test": examples[cut2:]}

    stats = {"records": n, "pairs": {}, "faults": Counter(), "clean_reps": 0, "paraphrases": para_stats}
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            count = 0
            for rid, task, obj, inp, refs in rows:
                for j, ref in enumerate(refs):
                    fh.write(json.dumps({
                        "id": rid, "task": task, "input": inp, "target": ref, "ref_index": j,
                        "record": obj.to_json_dict(),
                    }, ensure_ascii=False) + "\n")
                    count += 1
            stats["pairs"][name] = count
    for _, task, obj, _, _ in examples:
        if task == "rep":
            present = obj.present_faults(cfg)
            stats["faults"].update(f"{obj.exercise}:{c}" for c in present)
            stats["clean_reps"] += not present
    stats["faults"] = dict(sorted(stats["faults"].items()))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    # review sheet for the hand-verified test set
    test_rep = [e for e in splits["test"] if e[1] == "rep"][: args.verify_rep]
    test_ses = [e for e in splits["test"] if e[1] == "session"][: args.verify_session]
    with open(out / "test_to_verify.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "task", "input", "draft_reference", "verified_reference", "reviewed", "notes"])
        for rid, task, _, inp, refs in test_rep + test_ses:
            w.writerow([rid, task, inp, refs[0], refs[0], "", ""])

    # sheet for evaluating translation quality
    seen, rows = set(), []
    for rid, task, _, _, refs in splits["test"]:
        if refs[0] not in seen:
            seen.add(refs[0])
            rows.append((rid, task, refs[0]))
        if len(rows) >= args.translation_n:
            break
    with open(out / "translation_eval.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "task", "en", "ur_reference"])
        for r in rows:
            w.writerow([*r, ""])

    print(json.dumps({k: stats[k] for k in ("records", "pairs", "clean_reps", "paraphrases")}, indent=2))
    print(f"Wrote dataset to {out.resolve()}")


if __name__ == "__main__":
    main()
