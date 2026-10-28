"""Loads and validates the exercise configuration (config/exercises.yaml).

The config is the single source of truth for exercises, faults, templates and
fault-detection regexes. Everything else in the package reads it through
:func:`load_config`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "exercises.yaml"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    fixed: float | None = None
    relative_to: str | None = None
    normal: tuple[float, float] | None = None
    faults: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class FaultSpec:
    code: str
    label: str
    cue: dict[str, list[str]]
    praise: dict[str, list[str]]
    detect: tuple[re.Pattern, ...]
    praise_detect: tuple[re.Pattern, ...]
    summary: dict[str, str]
    summary_detect: tuple[re.Pattern, ...]
    tip: dict[str, str]
    deficit: dict[str, str] | None = None
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExerciseSpec:
    name: str
    display: dict[str, str]
    sided: bool
    limb: dict[str, str]
    features: dict[str, FeatureSpec]
    faults: dict[str, FaultSpec]  # exercise faults + generic faults, in config order

    @property
    def fault_codes(self) -> list[str]:
        return list(self.faults)


@dataclass(frozen=True)
class CoachConfig:
    fault_threshold: float
    max_corrections: int
    sides: dict[str, dict[str, str]]
    clean_rep: dict[str, list[str]]
    session_text: dict[str, Any]
    exercises: dict[str, ExerciseSpec]

    def exercise(self, name: str) -> ExerciseSpec:
        try:
            return self.exercises[name]
        except KeyError:
            known = ", ".join(sorted(self.exercises))
            raise ValueError(f"Unknown exercise '{name}'. Known exercises: {known}") from None


def _compile(patterns: list[str] | None) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in (patterns or []))


def _range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    lo, hi = value
    return float(lo), float(hi)


def _feature(name: str, raw: dict) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        fixed=raw.get("fixed"),
        relative_to=raw.get("relative_to"),
        normal=_range(raw.get("normal")),
        faults={code: _range(r) for code, r in (raw.get("faults") or {}).items()},
    )


def _fault(code: str, raw: dict) -> FaultSpec:
    return FaultSpec(
        code=code,
        label=raw.get("label", code.lower().replace("_", " ")),
        cue=raw["cue"],
        praise=raw.get("praise", {"en": [], "ur": []}),
        detect=_compile(raw.get("detect")),
        praise_detect=_compile(raw.get("praise_detect")),
        summary=raw["summary"],
        summary_detect=_compile(raw.get("summary_detect")),
        tip=raw["tip"],
        deficit=raw.get("deficit"),
        values=raw.get("values", {}),
    )


def parse_config(raw: dict) -> CoachConfig:
    generic = raw.get("generic", {})
    generic_features = {n: _feature(n, f) for n, f in generic.get("features", {}).items()}
    generic_faults = {c: _fault(c, f) for c, f in generic.get("faults", {}).items()}

    exercises: dict[str, ExerciseSpec] = {}
    for name, ex in raw["exercises"].items():
        features = {n: _feature(n, f) for n, f in ex["features"].items()}
        features.update(generic_features)
        faults = {c: _fault(c, f) for c, f in ex["faults"].items()}
        faults.update(generic_faults)
        for feat in features.values():
            unknown = set(feat.faults) - set(faults)
            if unknown:
                raise ValueError(f"{name}.{feat.name} references unknown faults {unknown}")
        exercises[name] = ExerciseSpec(
            name=name,
            display=ex["display"],
            sided=bool(ex.get("sided", False)),
            limb=ex.get("limb", {"en": "", "ur": ""}),
            features=features,
            faults=faults,
        )

    th = raw.get("thresholds", {})
    return CoachConfig(
        fault_threshold=float(th.get("fault_present", 0.5)),
        max_corrections=int(th.get("max_corrections_per_cue", 2)),
        sides=raw["sides"],
        clean_rep=raw["clean_rep"],
        session_text=raw["session_text"],
        exercises=exercises,
    )


@lru_cache(maxsize=8)
def _load(path: str) -> CoachConfig:
    with open(path, encoding="utf-8") as fh:
        return parse_config(yaml.safe_load(fh))


def load_config(path: str | os.PathLike | None = None) -> CoachConfig:
    """Load the config. ``PHYSIOCOACH_CONFIG`` overrides the bundled file."""
    path = path or os.environ.get("PHYSIOCOACH_CONFIG") or DEFAULT_CONFIG_PATH
    return _load(str(Path(path).resolve()))
