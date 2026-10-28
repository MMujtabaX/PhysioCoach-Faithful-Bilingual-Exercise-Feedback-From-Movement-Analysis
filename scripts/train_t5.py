"""Fine-tune Flan-T5 on the coaching dataset (Method B).

Run on Google Colab (free T4 GPU) or any CUDA machine:

    python scripts/train_t5.py --data data --model google/flan-t5-small --out models/flan-t5-coach

flan-t5-small (77M params) trains in roughly 15-25 minutes on a T4 with the
default dataset and runs fast on a laptop CPU. flan-t5-base (250M) is usually
more fluent but about 3x slower; try it if time allows and compare both.

After training, the script generates feedback for part of the validation set
and reports the faithfulness pass rate, so you know immediately whether the
model learned to state only true facts.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from physiocoach import RepRecord, SessionRecord, check_rep, check_session
from physiocoach.templates import all_rep_sentences
from physiocoach.tts import split_sentences


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def is_paraphrase(row: dict, known: set[str]) -> bool:
    """A rep target is a paraphrase if any sentence is not one the templates can produce."""
    return row["task"] == "rep" and any(s not in known for s in split_sentences(row["target"]))


def upweight_paraphrases(rows: list[dict], weight: int) -> tuple[list[dict], int]:
    """Repeat paraphrase pairs ``weight`` times in total (1 = unchanged)."""
    known = set(all_rep_sentences(lang="en", max_deficit=90))
    paras = [r for r in rows if is_paraphrase(r, known)]
    return rows + paras * max(0, weight - 1), len(paras)


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, max_in: int, max_out: int):
        self.rows, self.tok, self.max_in, self.max_out = rows, tokenizer, max_in, max_out

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["input"], max_length=self.max_in, truncation=True)
        enc["labels"] = self.tok(text_target=r["target"], max_length=self.max_out, truncation=True)["input_ids"]
        return enc


def _kwargs_supported(cls, **kwargs):
    """Map arguments across transformers versions (eval_strategy vs evaluation_strategy, ...)."""
    params = inspect.signature(cls.__init__).parameters
    out = {}
    for k, v in kwargs.items():
        if k in params:
            out[k] = v
        elif k == "eval_strategy" and "evaluation_strategy" in params:
            out["evaluation_strategy"] = v
        elif k == "processing_class" and "tokenizer" in params:
            out["tokenizer"] = v
    return out


def validate_faithfulness(model, tok, rows, device, batch_size=32):
    """Generate for unique validation records and measure the checker pass rate."""
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    model.eval()
    passed, examples, by_task = 0, [], {"rep": [0, 0], "session": [0, 0]}
    for i in range(0, len(uniq), batch_size):
        batch = uniq[i : i + batch_size]
        enc = tok([r["input"] for r in batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=384).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=160, num_beams=4)
        for r, ids in zip(batch, out):
            text = tok.decode(ids, skip_special_tokens=True).strip()
            obj = RepRecord(**r["record"]) if r["task"] == "rep" else SessionRecord(**r["record"])
            res = check_rep(text, obj) if r["task"] == "rep" else check_session(text, obj)
            passed += res.faithful
            by_task[r["task"]][0] += res.faithful
            by_task[r["task"]][1] += 1
            if len(examples) < 12 or not res.faithful and len(examples) < 30:
                examples.append({"input": r["input"], "output": text, "faithful": res.faithful, "issues": res.issues})
    return {
        "n": len(uniq),
        "pass_rate": passed / max(1, len(uniq)),
        "by_task": {k: (v[0] / v[1] if v[1] else None) for k, v in by_task.items()},
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="google/flan-t5-small")
    ap.add_argument("--out", default="models/flan-t5-coach")
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-in", type=int, default=384)
    ap.add_argument("--max-out", type=int, default=160)
    ap.add_argument("--max-train", type=int, default=None, help="cap training pairs (quick runs)")
    ap.add_argument("--val-check", type=int, default=400, help="validation records to faithfulness-check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--paraphrase-weight", type=int, default=1,
                    help="repeat LLM-paraphrased training pairs this many times (e.g. 5)")
    args = ap.parse_args()

    set_seed(args.seed)
    data = Path(args.data)
    out = Path(args.out)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  model: {args.model}")
    if device == "cpu":
        print("WARNING: training on CPU is very slow. Use Google Colab with a GPU (see the build guide).")

    # Native bf16 needs compute capability >= 8.0 (A100, L4, RTX 30xx+). On a T4 the
    # torch check can report True via emulation, which makes the Trainer refuse to start.
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    print(f"Precision: {'bf16' if use_bf16 else 'fp32'}")

    train_rows = read_jsonl(data / "train.jsonl", args.max_train)
    n_before = len(train_rows)
    train_rows, n_para = upweight_paraphrases(train_rows, args.paraphrase_weight)
    print(f"Paraphrase pairs: {n_para} of {n_before} ({100 * n_para / max(1, n_before):.1f}%); "
          f"weight {args.paraphrase_weight} -> {100 * n_para * args.paraphrase_weight / max(1, len(train_rows)):.1f}% "
          f"of {len(train_rows)} training pairs")
    val_rows = read_jsonl(data / "val.jsonl")
    train_ds = PairDataset(train_rows, tok, args.max_in, args.max_out)
    val_ds = PairDataset(val_rows[:2000], tok, args.max_in, args.max_out)
    print(f"Train pairs: {len(train_ds)}  |  val pairs (loss): {len(val_ds)}")

    targs = Seq2SeqTrainingArguments(**_kwargs_supported(
        Seq2SeqTrainingArguments,
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        bf16=use_bf16,  # T5 is unstable in fp16; T4 (Colab free) has no native bf16 -> fp32
        report_to="none",
        seed=args.seed,
    ))
    trainer = Seq2SeqTrainer(**_kwargs_supported(
        Seq2SeqTrainer,
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        processing_class=tok,
    ))

    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    # Some transformers versions save a T5 tokenizer that cannot be loaded back
    # (no spiece.model). Fine-tuning never changes the vocabulary, so if the saved
    # copy fails to load, save the base model's tokenizer instead.
    try:
        AutoTokenizer.from_pretrained(str(out))
    except Exception as e:
        print(f"Saved tokenizer did not reload ({type(e).__name__}); saving the base tokenizer instead")
        AutoTokenizer.from_pretrained(args.model).save_pretrained(str(out))
        AutoTokenizer.from_pretrained(str(out))
    print("Tokenizer reload check: OK")

    print("Checking faithfulness on validation records...")
    report = validate_faithfulness(trainer.model, tok, val_rows[: args.val_check * 3], device)
    report.update({"base_model": args.model, "epochs": args.epochs, "train_pairs": len(train_ds),
                   "paraphrase_pairs": n_para, "paraphrase_weight": args.paraphrase_weight,
                   "train_minutes": round(train_minutes, 1), "eval_loss": trainer.evaluate()["eval_loss"]})
    (out / "val_faithfulness.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("n", "pass_rate", "by_task", "eval_loss", "train_minutes")}, indent=2))
    print(f"Saved model to {out.resolve()}")


if __name__ == "__main__":
    main()
