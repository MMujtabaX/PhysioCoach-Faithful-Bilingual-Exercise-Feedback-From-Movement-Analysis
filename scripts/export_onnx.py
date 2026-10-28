"""Export a fine-tuned Flan-T5 to ONNX (optionally int8-quantized) for faster CPU inference.

    pip install -e ".[onnx]"
    python scripts/export_onnx.py --model models/flan-t5-coach-para-w5
    python scripts/export_onnx.py --model models/flan-t5-coach-para-w5 --quantize

Writes ``<model>-onnx`` (and ``<model>-onnx-int8`` with --quantize). Point the
app at it with PHYSIOCOACH_T5_DIR, then measure the speed-up with
scripts/benchmark_latency.py. Quantization makes the model smaller and usually
faster on CPU, but can change outputs slightly: always re-run
scripts/evaluate.py on the quantized model before using it.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def export(model_dir: Path, out: Path) -> Path:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    model = ORTModelForSeq2SeqLM.from_pretrained(model_dir, export=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(model_dir).save_pretrained(out)
    _copy_info(model_dir, out)
    print(f"Exported ONNX model to {out} in {time.perf_counter() - t0:.0f} s")
    return out


def quantize(onnx_dir: Path, out: Path) -> Path:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in onnx_dir.glob("*.onnx"))
    for name in files:
        ORTQuantizer.from_pretrained(onnx_dir, file_name=name).quantize(
            save_dir=out, quantization_config=qconfig, file_suffix="")
    # configs, generation config and tokenizer files (copied as files, never re-saved)
    for p in onnx_dir.iterdir():
        if p.suffix in {".json", ".txt", ".model"} and not (out / p.name).exists():
            shutil.copy(p, out / p.name)
    AutoTokenizer.from_pretrained(out)  # fail here, not later in the app, if something is missing
    print(f"Quantized {len(files)} ONNX files to int8 in {out}")
    return out


def _copy_info(src: Path, dst: Path) -> None:
    info = src / "val_faithfulness.json"   # lets the app find the base tokenizer if needed
    if info.exists():
        shutil.copy(info, dst / info.name)


def size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and "checkpoints" not in p.parts) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/flan-t5-coach-para-w5")
    ap.add_argument("--out", default=None, help="default: <model>-onnx")
    ap.add_argument("--quantize", action="store_true", help="also write an int8-quantized copy")
    args = ap.parse_args()

    model_dir = Path(args.model)
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"No model at {model_dir} (expected {model_dir / 'config.json'})")
    out = Path(args.out) if args.out else model_dir.with_name(model_dir.name + "-onnx")
    onnx_dir = export(model_dir, out)
    print(f"  PyTorch model: {size_mb(model_dir):.0f} MB   ONNX: {size_mb(onnx_dir):.0f} MB")
    if args.quantize:
        q = quantize(onnx_dir, onnx_dir.with_name(onnx_dir.name + "-int8"))
        print(f"  int8 ONNX: {size_mb(q):.0f} MB")
    print("\nNext: python scripts/benchmark_latency.py --models", model_dir, onnx_dir,
          *( [onnx_dir.with_name(onnx_dir.name + "-int8")] if args.quantize else []))


if __name__ == "__main__":
    main()
