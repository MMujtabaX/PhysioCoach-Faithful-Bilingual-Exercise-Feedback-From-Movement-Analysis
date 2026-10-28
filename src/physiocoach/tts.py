"""Text-to-speech for English and Urdu.

Default: Meta MMS-TTS (VITS) through Hugging Face Transformers, one small model
per language. MMS voices cannot read digits, so numbers are spelled out first.
Licence: MMS models are CC-BY-NC 4.0 (fine for coursework, not for commercial use).

Audio is cached by (language, text): coaching cues repeat a lot, so after a
short warm-up most cues play instantly.
"""

from __future__ import annotations

import hashlib
import re
import threading
import wave
from pathlib import Path

from .numbers import spell_digits

MMS_MODELS = {
    "en": "facebook/mms-tts-eng",
    "ur": "facebook/mms-tts-urd-script_arabic",
}


class Speaker:
    name = "base"

    def __init__(self, cache_dir: str | Path = "cache/audio"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, text: str, lang: str, kind: str = "cue") -> Path:
        h = hashlib.sha1(f"{self.name}|{lang}|{text}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{kind}_{lang}_{h}.wav"

    def _synthesize(self, text: str, lang: str, path: Path) -> None:  # pragma: no cover
        raise NotImplementedError

    def warmup(self, langs=("en", "ur")) -> None:
        """Load voices ahead of time so the first real request is not slow."""
        for lang in langs:
            tmp = self.cache_dir / f"_warmup_{lang}.wav"
            self._synthesize("ok" if lang == "en" else "ٹھیک", lang, tmp)
            tmp.unlink(missing_ok=True)

    def speak_sentence(self, sentence: str, lang: str) -> tuple[Path, bool]:
        """Synthesize (or fetch from cache) one sentence."""
        path = self.cache_path(sentence, lang, kind="sent")
        if path.exists():
            return path, True
        self._synthesize(spell_digits(sentence, lang), lang, path)
        return path, False

    def speak(self, text: str, lang: str) -> tuple[Path, bool]:
        """Return (wav_path, fully_cached).

        Audio is cached per *sentence* and the cue is assembled from those clips.
        Coaching cues are built from a small set of sentences, so once they are
        cached (see scripts/precompute_audio.py) any new cue plays instantly.
        """
        path = self.cache_path(text, lang)
        if path.exists():
            return path, True
        sentences = split_sentences(text)
        clips, cached = [], True
        for s in sentences:
            clip, hit = self.speak_sentence(s, lang)
            clips.append(clip)
            cached = cached and hit
        if len(clips) == 1:
            path.write_bytes(clips[0].read_bytes())
        else:
            concat_wavs(clips, path)
        return path, cached


_SENTENCE_RE = re.compile(r"(?<=[.!?۔])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def concat_wavs(paths: list[Path], out: Path, gap_s: float = 0.18) -> None:
    """Join WAV clips (same format) with a short pause between them."""
    params, frames = None, []
    for p in paths:
        with wave.open(str(p), "rb") as wf:
            if params is None:
                params = wf.getparams()
            elif (wf.getframerate(), wf.getsampwidth(), wf.getnchannels()) != (
                params.framerate, params.sampwidth, params.nchannels):
                raise ValueError(f"cannot join clips with different formats: {p}")
            frames.append(wf.readframes(wf.getnframes()))
    silence = b"\x00" * int(params.framerate * gap_s) * params.sampwidth * params.nchannels
    tmp = out.with_suffix(".tmp")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(params.nchannels)
        wf.setsampwidth(params.sampwidth)
        wf.setframerate(params.framerate)
        wf.writeframes(silence.join(frames))
    tmp.replace(out)


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    import numpy as np

    audio = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    tmp = path.with_suffix(".tmp")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    tmp.replace(path)


class MMSSpeaker(Speaker):
    name = "mms"

    def __init__(self, cache_dir: str | Path = "cache/audio", models: dict[str, str] | None = None,
                 device: str | None = None):
        super().__init__(cache_dir)
        self.models_ids = models or MMS_MODELS
        self._loaded: dict[str, tuple] = {}
        self._lock = threading.Lock()
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _get(self, lang: str):
        with self._lock:
            if lang not in self._loaded:
                from transformers import AutoTokenizer, VitsModel

                model_id = self.models_ids[lang]
                tok = AutoTokenizer.from_pretrained(model_id)
                model = VitsModel.from_pretrained(model_id).to(self.device).eval()
                self._loaded[lang] = (tok, model)
            return self._loaded[lang]

    def _synthesize(self, text: str, lang: str, path: Path) -> None:
        import torch

        tok, model = self._get(lang)
        if getattr(tok, "is_uroman", False):  # some MMS voices expect romanized input
            import uroman  # pip install uroman

            text = uroman.Uroman().romanize_string(text)
        inputs = tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            wav = model(**inputs).waveform[0].cpu().numpy()
        _write_wav(path, wav, model.config.sampling_rate)


class PiperSpeaker(Speaker):
    """Optional fast English voice (pip install piper-tts; download an .onnx voice)."""

    name = "piper"

    def __init__(self, voice_path: str | Path, cache_dir: str | Path = "cache/audio",
                 fallback: Speaker | None = None):
        super().__init__(cache_dir)
        from piper import PiperVoice

        self.voice = PiperVoice.load(str(voice_path))
        self.fallback = fallback  # used for Urdu

    def _synthesize(self, text: str, lang: str, path: Path) -> None:
        if lang != "en":
            if not self.fallback:
                raise ValueError("Piper voice is English only; configure an Urdu fallback speaker")
            self.fallback._synthesize(text, lang, path)
            return
        with wave.open(str(path), "wb") as wf:
            if hasattr(self.voice, "synthesize_wav"):   # piper-tts >= 1.3
                self.voice.synthesize_wav(text, wf)
            else:                                       # older piper-tts
                self.voice.synthesize(text, wf)
