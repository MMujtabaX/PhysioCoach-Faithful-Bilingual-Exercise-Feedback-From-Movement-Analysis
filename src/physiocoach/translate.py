"""English -> Urdu translation with a preservation check.

Urdu output is produced by translating the *already checked* English feedback,
so no Urdu training data or second generator is needed. A translation is only
accepted if it keeps every number and the left/right side; otherwise the
pipeline falls back to the hand-written Urdu template with the same content.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from .numbers import extract_numbers_en, extract_numbers_ur

UR_LEFT = ("بائیں", "بایاں", "بائیاں")
UR_RIGHT = ("دائیں", "دایاں", "دائیاں")
_EN_LEFT = re.compile(r"\bleft\b", re.IGNORECASE)
_EN_RIGHT = re.compile(r"\bright\b", re.IGNORECASE)
_EN_RIGHT_IDIOM = re.compile(r"(exactly|just|that's|thats|all) right\b", re.IGNORECASE)


@dataclass
class PreservationResult:
    ok: bool
    missing_numbers: list[int]
    extra_numbers: list[int]
    side_ok: bool

    def to_dict(self) -> dict:
        return {"ok": self.ok, "missing_numbers": self.missing_numbers,
                "extra_numbers": self.extra_numbers, "side_ok": self.side_ok}


def check_preservation(english: str, urdu: str) -> PreservationResult:
    en_nums = set(extract_numbers_en(english, include_words=False))
    ur_nums = set(extract_numbers_ur(urdu))
    missing = sorted(en_nums - ur_nums)
    extra = sorted(ur_nums - en_nums)

    en = _EN_RIGHT_IDIOM.sub(" ", english)
    has_left, has_right = bool(_EN_LEFT.search(en)), bool(_EN_RIGHT.search(en))
    ur_left = any(w in urdu for w in UR_LEFT)
    ur_right = any(w in urdu for w in UR_RIGHT)
    side_ok = has_left == ur_left and has_right == ur_right
    return PreservationResult(not missing and not extra and side_ok, missing, extra, side_ok)


class Translator:
    """Base class: caching + glossary post-editing around a ``_translate`` method."""

    name = "base"

    def __init__(self, cache_path: str | Path | None = None, glossary: dict[str, str] | None = None):
        self.cache_path = Path(cache_path) if cache_path else None
        self.glossary = glossary or {}
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        if self.cache_path and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _translate(self, text: str) -> str:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def apply_glossary(self, text: str) -> str:
        for bad, good in self.glossary.items():
            text = text.replace(bad, good)
        return text

    def translate(self, text: str) -> tuple[str, bool]:
        """Return (urdu_text, from_cache)."""
        with self._lock:
            if text in self._cache:
                return self._cache[text], True
        out = self.apply_glossary(self._translate(text))
        with self._lock:
            self._cache[text] = out
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return out, False


class NLLBTranslator(Translator):
    """Meta NLLB-200 (open weights, offline). Licence: CC-BY-NC 4.0 (non-commercial)."""

    name = "nllb"

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: str | None = None,
                 cache_path: str | Path | None = "cache/translations.json",
                 glossary: dict[str, str] | None = None, num_beams: int = 4):
        super().__init__(cache_path, glossary)
        self.num_beams = num_beams
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
        self._urd = self.tokenizer.convert_tokens_to_ids("urd_Arab")

    def _translate_one(self, sentence: str) -> str:
        import torch

        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, forced_bos_token_id=self._urd, max_new_tokens=128, num_beams=self.num_beams)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _translate(self, text: str) -> str:
        # NLLB is trained on single sentences; translate sentence by sentence.
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        return " ".join(self._translate_one(s) for s in sentences)


def load_glossary(path: str | Path) -> dict[str, str]:
    """Glossary file: JSON object mapping machine wording -> preferred wording."""
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}
