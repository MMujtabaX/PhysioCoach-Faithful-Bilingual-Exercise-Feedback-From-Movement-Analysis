"""Builds generators, the translator and the speaker from settings, lazily.

Heavy models load on first use, and a component that cannot load (missing
model, missing package, Ollama not running) reports *why* instead of crashing
the service.

Settings come from environment variables (or a ``.env``-style export):

=============================  ===============================  ==============
Variable                       Meaning                          Default
=============================  ===============================  ==============
PHYSIOCOACH_T5_DIR             fine-tuned Flan-T5 (or ONNX)     models/flan-t5-coach
PHYSIOCOACH_T5_SAMPLES         candidates for t5-guided         6
PHYSIOCOACH_T5_BEAMS           1 = greedy (fast), 4 = beam      1
PHYSIOCOACH_OLLAMA_MODEL       Ollama model tag                 qwen2.5:3b
PHYSIOCOACH_OLLAMA_HOST        Ollama server URL                http://localhost:11434
PHYSIOCOACH_TRANSLATOR         nllb | none                      nllb
PHYSIOCOACH_NLLB_MODEL         translation model id             facebook/nllb-200-distilled-600M
PHYSIOCOACH_NLLB_BEAMS         beam size (1 = fastest)          2
PHYSIOCOACH_WARMUP             load models at server start      1
PHYSIOCOACH_TTS                mms | piper | none               mms
PHYSIOCOACH_PIPER_VOICE        path to a Piper .onnx voice      (unset)
PHYSIOCOACH_CACHE_DIR          translation and audio cache      cache
PHYSIOCOACH_GLOSSARY           Urdu glossary JSON               glossary_ur.json
=============================  ===============================  ==============
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import CoachConfig, load_config
from .generators import BackendUnavailable, GuidedT5Backend, OllamaBackend, T5Backend, TemplateBackend
from .pipeline import CoachPipeline


@dataclass
class Settings:
    t5_dir: str = "models/flan-t5-coach"
    t5_samples: int = 6
    t5_beams: int = 1
    ollama_model: str = "qwen2.5:3b"
    ollama_host: str = "http://localhost:11434"
    translator: str = "nllb"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    nllb_beams: int = 2
    warmup: bool = True
    tts: str = "mms"
    piper_voice: str | None = None
    cache_dir: str = "cache"
    glossary: str = "glossary_ur.json"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            t5_dir=env("PHYSIOCOACH_T5_DIR", cls.t5_dir),
            t5_samples=int(env("PHYSIOCOACH_T5_SAMPLES", cls.t5_samples)),
            t5_beams=int(env("PHYSIOCOACH_T5_BEAMS", cls.t5_beams)),
            ollama_model=env("PHYSIOCOACH_OLLAMA_MODEL", cls.ollama_model),
            ollama_host=env("PHYSIOCOACH_OLLAMA_HOST", cls.ollama_host),
            translator=env("PHYSIOCOACH_TRANSLATOR", cls.translator).lower(),
            nllb_model=env("PHYSIOCOACH_NLLB_MODEL", cls.nllb_model),
            nllb_beams=int(env("PHYSIOCOACH_NLLB_BEAMS", cls.nllb_beams)),
            warmup=env("PHYSIOCOACH_WARMUP", "1") not in ("0", "false", "no"),
            tts=env("PHYSIOCOACH_TTS", cls.tts).lower(),
            piper_voice=env("PHYSIOCOACH_PIPER_VOICE"),
            cache_dir=env("PHYSIOCOACH_CACHE_DIR", cls.cache_dir),
            glossary=env("PHYSIOCOACH_GLOSSARY", cls.glossary),
        )


@dataclass
class Registry:
    settings: Settings = field(default_factory=Settings.from_env)
    cfg: CoachConfig = field(default_factory=load_config)

    def __post_init__(self):
        self._lock = threading.Lock()
        self._generators: dict = {}
        self._errors: dict[str, str] = {}
        self._translator = None
        self._translator_tried = False
        self._speaker = None
        self._speaker_tried = False
        self.warmed_up: dict | None = None

    # ------------------------------------------------------------ generators
    def generator(self, name: str):
        with self._lock:
            if name in self._generators:
                return self._generators[name]
            s = self.settings
            try:
                if name == "template":
                    gen = TemplateBackend(self.cfg)
                elif name == "t5":
                    gen = T5Backend(s.t5_dir, num_beams=s.t5_beams, cfg=self.cfg)
                elif name == "t5-guided":
                    gen = GuidedT5Backend(s.t5_dir, n_samples=s.t5_samples, num_beams=s.t5_beams, cfg=self.cfg)
                elif name == "llm":
                    if not OllamaBackend.is_running(s.ollama_host):
                        raise BackendUnavailable(f"Ollama is not running at {s.ollama_host}")
                    gen = OllamaBackend(s.ollama_model, s.ollama_host, cfg=self.cfg)
                else:
                    raise BackendUnavailable(f"unknown backend '{name}' (use template, t5, t5-guided or llm)")
            except BackendUnavailable as e:
                self._errors[name] = str(e)
                raise
            self._generators[name] = gen
            self._errors.pop(name, None)
            return gen

    # ------------------------------------------------------------ translator
    @property
    def translator(self):
        with self._lock:
            if not self._translator_tried:
                self._translator_tried = True
                if self.settings.translator == "nllb":
                    try:
                        from .translate import NLLBTranslator, load_glossary

                        self._translator = NLLBTranslator(
                            self.settings.nllb_model,
                            cache_path=Path(self.settings.cache_dir) / "translations.json",
                            glossary=load_glossary(self.settings.glossary),
                            num_beams=self.settings.nllb_beams,
                        )
                    except Exception as e:  # missing extras or download failure
                        self._errors["translator"] = f"{type(e).__name__}: {e}"
            return self._translator

    # --------------------------------------------------------------- speaker
    @property
    def speaker(self):
        with self._lock:
            if not self._speaker_tried:
                self._speaker_tried = True
                audio_dir = Path(self.settings.cache_dir) / "audio"
                try:
                    if self.settings.tts == "mms":
                        from .tts import MMSSpeaker

                        self._speaker = MMSSpeaker(audio_dir)
                    elif self.settings.tts == "piper":
                        from .tts import MMSSpeaker, PiperSpeaker

                        if not self.settings.piper_voice:
                            raise ValueError("set PHYSIOCOACH_PIPER_VOICE to a Piper .onnx voice file")
                        self._speaker = PiperSpeaker(self.settings.piper_voice, audio_dir,
                                                     fallback=MMSSpeaker(audio_dir))
                except Exception as e:
                    self._errors["tts"] = f"{type(e).__name__}: {e}"
            return self._speaker

    # ---------------------------------------------------------------- warmup
    def warmup(self) -> dict:
        """Load the translator and voices now (called in the background at server start)."""
        import time

        done = {}
        t0 = time.perf_counter()
        if self.speaker is not None:
            try:
                self.speaker.warmup()
                done["tts_s"] = round(time.perf_counter() - t0, 1)
            except Exception as e:
                self._errors["tts"] = f"warmup failed: {type(e).__name__}: {e}"
        t0 = time.perf_counter()
        if self.translator is not None:
            try:
                self.translator._translate("Good.")
                done["translator_s"] = round(time.perf_counter() - t0, 1)
            except Exception as e:
                self._errors["translator"] = f"warmup failed: {type(e).__name__}: {e}"
        self.warmed_up = done
        return done

    # ------------------------------------------------------------- helpers
    def pipeline(self, backend: str, need_translation: bool, need_speech: bool, check: bool = True):
        return CoachPipeline(
            self.generator(backend),
            translator=self.translator if need_translation else None,
            speaker=self.speaker if need_speech else None,
            check=check,
            cfg=self.cfg,
        )

    def status(self) -> dict:
        s = self.settings
        return {
            "backends": {
                "template": {"available": True},
                "t5": {"available": Path(s.t5_dir).exists(), "model_dir": s.t5_dir,
                       "error": self._errors.get("t5")},
                "t5-guided": {"available": Path(s.t5_dir).exists(), "samples": s.t5_samples,
                              "error": self._errors.get("t5-guided")},
                "llm": {"available": OllamaBackend.is_running(s.ollama_host), "model": s.ollama_model,
                        "error": self._errors.get("llm")},
            },
            "translator": {"configured": s.translator, "loaded": self._translator is not None,
                           "error": self._errors.get("translator")},
            "tts": {"configured": s.tts, "loaded": self._speaker is not None, "error": self._errors.get("tts")},
            "warmed_up": self.warmed_up,
        }
