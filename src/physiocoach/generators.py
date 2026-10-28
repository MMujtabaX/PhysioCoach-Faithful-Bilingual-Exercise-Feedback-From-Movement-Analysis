"""The three generation methods compared in the project.

* :class:`TemplateBackend` (A): rule-based baseline, always faithful.
* :class:`T5Backend` (B): fine-tuned Flan-T5 on linearized records (main system).
* :class:`OllamaBackend` (C): few-shot prompting of a local open-weight LLM.

All three produce *English*; Urdu comes from translation in the pipeline.
Heavy libraries are imported lazily so the template backend runs with no ML
dependencies installed.
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from .config import CoachConfig, load_config
from .linearize import linearize_rep, linearize_session
from .records import random_rep, random_session
from .schema import RepRecord, SessionRecord
from .templates import TemplateGenerator


class Generator(Protocol):
    name: str

    def rep(self, record: RepRecord) -> str: ...

    def session(self, session: SessionRecord) -> str: ...


class BackendUnavailable(RuntimeError):
    """Raised when a backend's model or service cannot be loaded."""


# --------------------------------------------------------------------- A
class TemplateBackend:
    name = "template"

    def __init__(self, cfg: CoachConfig | None = None, seed: int = 0):
        self.gen = TemplateGenerator(cfg, seed=seed)

    def rep(self, record: RepRecord) -> str:
        return self.gen.rep(record, "en")

    def session(self, session: SessionRecord) -> str:
        return self.gen.session(session, "en")


# --------------------------------------------------------------------- B
def _load_tokenizer(model_dir: Path):
    """Load the saved tokenizer; if it is broken, use the base model's (same vocabulary)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        base = "google/flan-t5-small"
        info = model_dir / "val_faithfulness.json"
        if info.exists():
            base = json.loads(info.read_text()).get("base_model", base)
        tok = AutoTokenizer.from_pretrained(base)
        try:
            tok.save_pretrained(model_dir)  # repair the folder for next time
        except OSError:
            pass
        return tok


class T5Backend:
    """Fine-tuned Flan-T5. Loads a PyTorch checkpoint or an ONNX export."""

    name = "t5"

    def __init__(self, model_dir: str | Path, device: str | None = None, num_beams: int = 1,
                 cfg: CoachConfig | None = None):
        self.cfg = cfg or load_config()
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise BackendUnavailable(
                f"No fine-tuned model at {model_dir}. Train one with scripts/train_t5.py "
                "(see docs/BUILD_GUIDE.md, step 5)."
            )
        try:
            import transformers  # noqa: F401  (check the ML extras are installed)
        except ImportError as e:  # pragma: no cover
            raise BackendUnavailable("Install ML extras: pip install -e '.[ml]'") from e

        self.tokenizer = _load_tokenizer(model_dir)
        self.num_beams = num_beams
        self.onnx = any(model_dir.glob("*encoder*.onnx"))
        if self.onnx:
            try:
                from optimum.onnxruntime import ORTModelForSeq2SeqLM
            except ImportError as e:  # pragma: no cover
                raise BackendUnavailable("ONNX model found; install: pip install -e '.[onnx]'") from e
            self.model = ORTModelForSeq2SeqLM.from_pretrained(model_dir)
            self.device = "cpu"
        else:
            import torch
            from transformers import AutoModelForSeq2SeqLM

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device).eval()

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
        if not self.onnx:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            import torch

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=self.num_beams)
        else:
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=self.num_beams)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def rep(self, record: RepRecord) -> str:
        return self._generate(linearize_rep(record, self.cfg), 48)

    def session(self, session: SessionRecord) -> str:
        return self._generate(linearize_session(session, self.cfg), 160)

    def _sample(self, prompt: str, max_new_tokens: int, n: int, top_p: float, temperature: float) -> list[str]:
        """n sampled candidates plus the beam-search output, decoded."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
                      temperature=temperature, num_return_sequences=n)
        if not self.onnx:
            import torch

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(**inputs, **kwargs)
        else:
            out = self.model.generate(**inputs, **kwargs)
        sampled = [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in out]
        return sampled


class GuidedT5Backend(T5Backend):
    """Flan-T5 with *faithfulness-constrained sampling* (checker-guided decoding).

    Beam search always returns the single most likely sentence, so a fine-tuned
    model collapses onto one favourite phrasing per fault. Here the model samples
    several candidates, the faithfulness checker removes every candidate that
    states something the input does not support, and one of the remaining
    distinct candidates is chosen at random. The beam output is always among the
    candidates, so this is never less faithful than plain Flan-T5.

    ``last_stats`` records how many candidates were generated and passed, which
    the evaluation reports as the checker's rejection rate during decoding.
    """

    name = "t5-guided"

    def __init__(self, model_dir: str | Path, n_samples: int = 6, top_p: float = 0.92,
                 temperature: float = 1.0, seed: int | None = None, **kwargs):
        super().__init__(model_dir, **kwargs)
        self.n_samples, self.top_p, self.temperature = n_samples, top_p, temperature
        self.rng = random.Random(seed)
        self.last_stats: dict = {}
        if seed is not None and not self.onnx:
            import torch

            torch.manual_seed(seed)

    def _choose(self, candidates: list[str], is_faithful, fallback) -> str:
        """Pick a random faithful candidate; only if none passes, decode deterministically.

        ``fallback`` (a greedy/beam decode) runs lazily, so the common case costs a
        single sampling call. The pipeline's template fallback still catches the rare
        case where the deterministic output fails too.
        """
        seen, faithful = set(), []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                if is_faithful(c):
                    faithful.append(c)
        self.last_stats = {"candidates": len(candidates), "distinct": len(seen), "faithful": len(faithful),
                           "fallback_decode": not faithful}
        if faithful:
            return self.rng.choice(faithful)
        return fallback()

    def rep(self, record: RepRecord) -> str:
        from .faithfulness import check_rep

        prompt = linearize_rep(record, self.cfg)
        cands = self._sample(prompt, 48, self.n_samples, self.top_p, self.temperature)
        return self._choose(cands, lambda t: check_rep(t, record, self.cfg).faithful,
                            lambda: self._generate(prompt, 48))

    def session(self, session: SessionRecord) -> str:
        from .faithfulness import check_session

        prompt = linearize_session(session, self.cfg)
        cands = self._sample(prompt, 160, max(2, self.n_samples // 2), self.top_p, self.temperature)
        return self._choose(cands, lambda t: check_session(t, session, self.cfg).faithful,
                            lambda: self._generate(prompt, 160))


# --------------------------------------------------------------------- C
_SYSTEM = (
    "You are a physiotherapy exercise coach. You receive a structured analysis of one "
    "exercise repetition (or a whole session) and write short, encouraging feedback in "
    "plain English. Rules: mention ONLY faults listed under 'faults'; never correct anything "
    "listed under 'clear'; use only numbers that appear in the input; say left/right only if "
    "the input has a side; a rep cue is one or two short sentences; a session summary is at "
    "most five sentences. Output the feedback text only."
)


class OllamaBackend:
    """Few-shot LLM via a local Ollama server (https://ollama.com)."""

    name = "llm"

    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434",
                 n_shots: int = 4, temperature: float = 0.3, timeout: float = 120.0,
                 cfg: CoachConfig | None = None):
        self.cfg = cfg or load_config()
        self.model, self.host, self.temperature, self.timeout = model, host.rstrip("/"), temperature, timeout
        self._rep_shots, self._session_shots = self._build_shots(n_shots)

    @staticmethod
    def is_running(host: str = "http://localhost:11434", timeout: float = 0.5) -> bool:
        try:
            with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def _build_shots(self, n: int):
        rng = random.Random(1234)
        tmpl = TemplateGenerator(self.cfg, seed=99)
        reps = []
        for i in range(n):
            r = random_rep(rng, p_fault=0.0 if i == n - 1 else 0.45, cfg=self.cfg)
            reps.append((linearize_rep(r, self.cfg), tmpl.rep(r)))
        sessions = []
        for _ in range(2):
            s = random_session(rng, n_sets=2, cfg=self.cfg)
            sessions.append((linearize_session(s, self.cfg), tmpl.session(s)))
        return reps, sessions

    def _prompt(self, shots, query: str) -> str:
        lines = [_SYSTEM, ""]
        for inp, out in shots:
            lines += [f"Input: {inp}", f"Feedback: {out}", ""]
        lines += [f"Input: {query}", "Feedback:"]
        return "\n".join(lines)

    def _call(self, prompt: str, max_tokens: int) -> str:
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": max_tokens, "stop": ["\nInput:"]},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                text = json.loads(resp.read())["response"]
        except (urllib.error.URLError, OSError) as e:
            raise BackendUnavailable(f"Ollama not reachable at {self.host}. Run 'ollama serve' "
                                     f"and 'ollama pull {self.model}'.") from e
        return text.strip().split("\n\n")[0].strip().strip('"')

    def rep(self, record: RepRecord) -> str:
        return self._call(self._prompt(self._rep_shots, linearize_rep(record, self.cfg)), 60)

    def session(self, session: SessionRecord) -> str:
        return self._call(self._prompt(self._session_shots, linearize_session(session, self.cfg)), 200)

    def paraphrase(self, text: str) -> str:
        """Reword a cue while keeping its meaning (used to diversify the dataset)."""
        prompt = (
            "Rewrite this exercise coaching feedback in different words. Keep exactly the same "
            "meaning, the same numbers and the same left/right side. Keep it short and friendly. "
            f"Output only the rewritten feedback.\n\nFeedback: {text}\nRewritten:"
        )
        return self._call(prompt, 60)
