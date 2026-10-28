"""FastAPI service: the single entry point PhysioLens (or any client) calls.

Run:  uvicorn physiocoach.api:app --reload   (or: physiocoach serve)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .facts import rep_facts, session_facts
from .factory import Registry
from .faithfulness import annotate_rep, annotate_session, check_rep, check_session
from .generators import BackendUnavailable
from .records import random_rep, random_session
from .schema import RepRecord, SessionRecord

WEB_DIR = Path(__file__).parent / "web"

Backend = Literal["template", "t5", "t5-guided", "llm"]
Lang = Literal["en", "ur"]


class RepRequest(BaseModel):
    record: RepRecord
    lang: Lang = "en"
    backend: Backend = "template"
    speak: bool = False
    check: bool = Field(True, description="Disable only for the ablation study")
    live: bool = Field(True, description="Instant Urdu from hand-written templates instead of machine "
                                         "translation; use for cues heard mid-exercise")


class SessionRequest(BaseModel):
    session: SessionRecord
    lang: Lang = "en"
    backend: Backend = "template"
    speak: bool = False
    check: bool = True
    live: bool = False


class CompareRequest(BaseModel):
    record: RepRecord | None = None
    session: SessionRecord | None = None
    backends: list[Backend] = ["template", "t5", "t5-guided", "llm"]


def create_app(registry: Registry | None = None) -> FastAPI:
    reg = registry or Registry()

    @asynccontextmanager
    async def lifespan(app):
        if reg.settings.warmup and (reg.settings.tts != "none" or reg.settings.translator != "none"):
            # load models in the background so the server answers immediately
            threading.Thread(target=reg.warmup, daemon=True, name="warmup").start()
        yield

    app = FastAPI(
        lifespan=lifespan,
        title="PhysioCoach-NLP",
        version=__version__,
        description="Faithful exercise coaching feedback (English + Urdu, with speech) "
                    "from structured movement analysis.",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.registry = reg

    def _respond(fb):
        d = fb.to_dict()
        d["audio_url"] = f"/audio/{Path(fb.audio_path).name}" if fb.audio_path else None
        d.pop("audio_path", None)
        return d

    def _pipeline(backend: str, lang: str, speak: bool, check: bool):
        try:
            pipe = reg.pipeline(backend, lang == "ur", speak, check)
        except BackendUnavailable as e:
            raise HTTPException(503, str(e)) from None
        if speak and pipe.speaker is None:
            raise HTTPException(503, f"Text-to-speech is not available: {reg.status()['tts']['error'] or 'disabled'}")
        return pipe

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    # the old page addresses open the single-page app; app.js selects the tab (/#compare, /#live)
    @app.get("/compare", include_in_schema=False)
    def compare_page():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/live", include_in_schema=False)
    def live_page():
        return FileResponse(WEB_DIR / "index.html")

    def _facts(record: RepRecord | None, session: SessionRecord | None) -> dict:
        if record is not None:
            f = rep_facts(record, reg.cfg)
            ex = f.exercise
            return {"kind": "rep", "exercise": ex.display["en"], "side": f.side,
                    "present": [{"label": ex.faults[c].label, "p": record.faults[c], **f.values.get(c, {})}
                                for c in f.present],
                    "absent": [ex.faults[c].label for c in f.absent],
                    "required": [ex.faults[c].label for c in f.required]}
        f = session_facts(session, reg.cfg)
        ex = f.exercise
        return {"kind": "session", "exercise": ex.display["en"], "side": session.side,
                "total": f.stats.total, "sets": len(f.stats.sets),
                "counts": [{"label": ex.faults[c].label, "count": f.stats.fault_counts[c]} for c in f.occurred],
                "required": [ex.faults[c].label for c in f.notable],
                "fatigue": (f"{ex.faults[f.fatigue.fault].label}: {f.fatigue.last_count} reps in set "
                            f"{f.fatigue.last_set}") if f.fatigue else None}

    @app.post("/api/compare")
    def compare(req: CompareRequest):
        """Run several generators on one record and annotate every sentence (no fallback applied)."""
        if (req.record is None) == (req.session is None):
            raise HTTPException(422, "send exactly one of 'record' or 'session'")
        obj = req.record or req.session
        try:
            obj.validate_against(reg.cfg)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        is_rep = req.record is not None
        template_text = reg.generator("template").rep(obj) if is_rep else reg.generator("template").session(obj)

        def run(name: str) -> dict:
            try:
                gen = reg.generator(name)
            except BackendUnavailable as e:
                return {"backend": name, "available": False, "error": str(e)}
            t0 = time.perf_counter()
            try:
                text = gen.rep(obj) if is_rep else gen.session(obj)
            except BackendUnavailable as e:
                return {"backend": name, "available": False, "error": str(e)}
            ms = round((time.perf_counter() - t0) * 1000, 1)
            res = check_rep(text, obj, reg.cfg) if is_rep else check_session(text, obj, reg.cfg)
            ann = annotate_rep(text, obj, reg.cfg) if is_rep else annotate_session(text, obj, reg.cfg)
            return {"backend": name, "available": True, "text": text, "latency_ms": ms,
                    "faithful": res.faithful, "issues": res.issues, "annotation": ann,
                    "decoding": dict(getattr(gen, "last_stats", {}) or {}) if name == "t5-guided" else {},
                    "replacement": None if res.faithful else template_text}

        with ThreadPoolExecutor(max_workers=len(req.backends) or 1) as pool:
            results = list(pool.map(run, req.backends))
        return {"facts": _facts(req.record, req.session), "results": results}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/status")
    def status():
        return reg.status()

    @app.get("/api/exercises")
    def exercises():
        return {
            name: {"display": ex.display, "sided": ex.sided,
                   "faults": {c: f.label for c, f in ex.faults.items()},
                   "features": list(ex.features)}
            for name, ex in reg.cfg.exercises.items()
        }

    @app.get("/api/sample/rep")
    def sample_rep(exercise: str | None = None, p_fault: float = Query(0.35, ge=0, le=1),
                   seed: int | None = None):
        try:
            return random_rep(random.Random(seed), exercise, p_fault, reg.cfg).to_json_dict()
        except ValueError as e:
            raise HTTPException(422, str(e)) from None

    @app.get("/api/sample/session")
    def sample_session(exercise: str | None = None, sets: int = Query(2, ge=1, le=5),
                       reps_per_set: int = Query(10, ge=1, le=30), seed: int | None = None):
        try:
            return random_session(random.Random(seed), exercise, sets, reps_per_set, reg.cfg).to_json_dict()
        except ValueError as e:
            raise HTTPException(422, str(e)) from None

    @app.post("/api/feedback/rep")
    def feedback_rep(req: RepRequest):
        pipe = _pipeline(req.backend, req.lang, req.speak, req.check)
        try:
            return _respond(pipe.rep(req.record, req.lang, req.speak, req.live))
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        except BackendUnavailable as e:
            raise HTTPException(503, str(e)) from None

    @app.post("/api/feedback/session")
    def feedback_session(req: SessionRequest):
        pipe = _pipeline(req.backend, req.lang, req.speak, req.check)
        try:
            return _respond(pipe.session(req.session, req.lang, req.speak, req.live))
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        except BackendUnavailable as e:
            raise HTTPException(503, str(e)) from None

    @app.get("/audio/{name}", include_in_schema=False)
    def audio(name: str):
        audio_dir = (Path(reg.settings.cache_dir) / "audio").resolve()
        path = (audio_dir / name).resolve()
        if path.parent != audio_dir or not path.exists() or path.suffix != ".wav":
            raise HTTPException(404, "audio not found")
        return FileResponse(path, media_type="audio/wav")

    return app


app = create_app()
