"""Paraphrase detection used by --paraphrase-weight (no torch needed)."""
import importlib.util
import random
import sys
import types
from pathlib import Path

from physiocoach import TemplateGenerator
from physiocoach.records import random_rep


def load_train_module():
    # stub the heavy imports so the helper functions can be tested without torch
    for name in ("torch", "torch.utils", "torch.utils.data", "transformers"):
        mod = types.ModuleType(name)
        mod.Dataset = object
        for attr in ("AutoModelForSeq2SeqLM", "AutoTokenizer", "DataCollatorForSeq2Seq",
                     "Seq2SeqTrainer", "Seq2SeqTrainingArguments", "set_seed"):
            setattr(mod, attr, None)
        sys.modules.setdefault(name, mod)
    path = Path(__file__).parents[1] / "scripts" / "train_t5.py"
    spec = importlib.util.spec_from_file_location("train_t5", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_templates_are_not_paraphrases_and_rewrites_are():
    m = load_train_module()
    gen, rng = TemplateGenerator(), random.Random(0)
    rows = [{"task": "rep", "target": gen.rep(random_rep(rng, p_fault=0.5))} for _ in range(500)]
    rows += [{"task": "rep", "target": "Try lifting that leg a touch higher, you're about 12 degrees off."},
             {"task": "session", "target": "Something new entirely."}]
    out, n_para = m.upweight_paraphrases(rows, 5)
    assert n_para == 1                  # only the rep rewrite; sessions are never counted
    assert len(out) == len(rows) + 4
