"""End-to-end coaching pipeline.

    record -> generator (English) -> faithfulness check --fail--> English template
                                          |
                                         pass
                                          v
                       lang == "ur" ? translate -> preservation check --fail--> Urdu template
                                          v
                                    text-to-speech (optional)

Every step degrades gracefully: with no translator configured, Urdu comes from
the hand-written templates; with no speaker, no audio is produced.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from .config import CoachConfig, load_config
from .faithfulness import check_rep, check_session
from .generators import Generator
from .schema import RepRecord, SessionRecord
from .templates import TemplateGenerator
from .translate import Translator, check_preservation
from .tts import Speaker

Lang = Literal["en", "ur"]


@dataclass
class Feedback:
    kind: str
    lang: str
    backend: str
    text: str                        # final text in the requested language
    text_en: str                     # final English text (after any fallback)
    raw: str                         # what the generator produced before checking
    faithful_raw: bool               # did the generator's own output pass the checker?
    fallback: bool                   # was the English template used instead?
    check: dict
    translation: dict | None = None
    audio_path: str | None = None
    timings_ms: dict = field(default_factory=dict)
    decoding: dict = field(default_factory=dict)   # guided decoding: candidates / distinct / faithful

    def to_dict(self) -> dict:
        return asdict(self)


class CoachPipeline:
    def __init__(self, generator: Generator, translator: Translator | None = None,
                 speaker: Speaker | None = None, check: bool = True, cfg: CoachConfig | None = None):
        self.cfg = cfg or load_config()
        self.generator = generator
        self.translator = translator
        self.speaker = speaker
        self.check_enabled = check
        self.templates = TemplateGenerator(self.cfg)

    # ------------------------------------------------------------------ api
    def rep(self, record: RepRecord, lang: Lang = "en", speak: bool = False, live: bool = False) -> Feedback:
        """``live=True``: Urdu comes from the hand-written templates (instant, same facts)
        instead of machine translation, for cues the user hears mid-exercise."""
        record.validate_against(self.cfg)
        return self._run("rep", record, lang, speak, live)

    def session(self, session: SessionRecord, lang: Lang = "en", speak: bool = False, live: bool = False) -> Feedback:
        session.validate_against(self.cfg)
        return self._run("session", session, lang, speak, live)

    # -------------------------------------------------------------- internals
    def _run(self, kind: str, obj, lang: str, speak: bool, live: bool = False) -> Feedback:
        if lang not in ("en", "ur"):
            raise ValueError("lang must be 'en' or 'ur'")
        timings: dict[str, float] = {}
        is_rep = kind == "rep"

        t = time.perf_counter()
        raw = self.generator.rep(obj) if is_rep else self.generator.session(obj)
        timings["generate"] = _ms(t)
        decoding = dict(getattr(self.generator, "last_stats", {}) or {})

        t = time.perf_counter()
        result = check_rep(raw, obj, self.cfg) if is_rep else check_session(raw, obj, self.cfg)
        timings["check"] = _ms(t)

        text_en, fallback = raw, False
        if self.check_enabled and not result.faithful:
            text_en, fallback = self._template(kind, obj, "en"), True

        text, translation = text_en, None
        if lang == "ur":
            t = time.perf_counter()
            if live:
                text = self._template(kind, obj, "ur")
                translation = {"method": "template", "reason": "live mode: hand-written Urdu, no translation delay"}
            else:
                text, translation = self._to_urdu(kind, obj, text_en)
            timings["translate"] = _ms(t)

        audio = None
        if speak and self.speaker:
            t = time.perf_counter()
            path, cached = self.speaker.speak(text, lang)
            audio = str(path)
            timings["speak"] = _ms(t)
            timings["speak_cached"] = cached

        return Feedback(
            kind=kind, lang=lang, backend=getattr(self.generator, "name", "custom"),
            text=text, text_en=text_en, raw=raw, faithful_raw=result.faithful,
            fallback=fallback, check=result.to_dict(), translation=translation,
            audio_path=audio, timings_ms=timings, decoding=decoding,
        )

    def _template(self, kind: str, obj, lang: str) -> str:
        return self.templates.rep(obj, lang) if kind == "rep" else self.templates.session(obj, lang)

    def _to_urdu(self, kind: str, obj, text_en: str) -> tuple[str, dict]:
        if not self.translator:
            return self._template(kind, obj, "ur"), {"method": "template", "reason": "no translator configured"}
        urdu, cached = self.translator.translate(text_en)
        pres = check_preservation(text_en, urdu)
        if pres.ok:
            return urdu, {"method": self.translator.name, "cached": cached, "preservation": pres.to_dict(),
                          "fallback": False}
        return self._template(kind, obj, "ur"), {
            "method": "template", "machine_translation": urdu, "cached": cached,
            "preservation": pres.to_dict(), "fallback": True,
        }


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
