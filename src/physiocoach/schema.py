"""Input schema: one record per repetition, and a session made of repetitions.

This schema is the *only* contract between PhysioCoach-NLP and a movement
analysis system (for example the PhysioLens CNN-LSTM). Anything that can emit
these records can get coaching feedback.

Example rep record::

    {
      "exercise": "side_leg_raise",
      "side": "left",
      "set": 2,
      "rep": 7,
      "faults": {"SHALLOW": 0.91, "KNEE_BENT": 0.12, "TORSO_LEAN": 0.05, "TOO_FAST": 0.10},
      "features": {"peak_hip_abduction_deg": 28.0, "target_hip_abduction_deg": 40.0,
                   "knee_angle_deg": 172.0, "trunk_lean_deg": 4.0, "rep_duration_s": 2.1}
    }
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import CoachConfig, load_config

Side = Literal["left", "right"]


class RepRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    exercise: str
    side: Side | None = None
    set_index: int = Field(1, alias="set", ge=1)
    rep: int = Field(1, ge=1)
    faults: dict[str, float] = Field(default_factory=dict, description="Fault code -> probability (0-1)")
    features: dict[str, float] = Field(default_factory=dict)

    @field_validator("faults")
    @classmethod
    def _probabilities(cls, v: dict[str, float]) -> dict[str, float]:
        for code, p in v.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"fault probability for {code} must be in [0, 1], got {p}")
        return v

    def validate_against(self, cfg: CoachConfig | None = None) -> "RepRecord":
        cfg = cfg or load_config()
        ex = cfg.exercise(self.exercise)
        unknown = set(self.faults) - set(ex.faults)
        if unknown:
            raise ValueError(f"Unknown faults for {self.exercise}: {sorted(unknown)}")
        if ex.sided and self.side is None:
            raise ValueError(f"{self.exercise} is a one-sided exercise; 'side' is required")
        return self

    def present_faults(self, cfg: CoachConfig | None = None) -> list[str]:
        """Faults at or above the threshold, highest probability first."""
        cfg = cfg or load_config()
        present = [(p, c) for c, p in self.faults.items() if p >= cfg.fault_threshold]
        return [c for _, c in sorted(present, key=lambda t: (-t[0], t[1]))]

    def to_json_dict(self) -> dict:
        return self.model_dump(by_alias=True)


class SessionRecord(BaseModel):
    exercise: str
    side: Side | None = None
    reps: list[RepRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "SessionRecord":
        for r in self.reps:
            if r.exercise != self.exercise:
                raise ValueError("all reps in a session must share the session's exercise")
        return self

    def validate_against(self, cfg: CoachConfig | None = None) -> "SessionRecord":
        for r in self.reps:
            r.validate_against(cfg)
        return self

    def to_json_dict(self) -> dict:
        return self.model_dump(by_alias=True)


class SessionStats(BaseModel):
    """Aggregates computed from a session; used by templates, checker and linearizer."""

    exercise: str
    side: Side | None
    total: int
    sets: list[int]                              # set indices in order
    reps_per_set: dict[int, int]
    fault_counts: dict[str, int]                 # total per fault (only faults that occurred)
    fault_counts_by_set: dict[int, dict[str, int]]

    @classmethod
    def from_session(cls, s: SessionRecord, cfg: CoachConfig | None = None) -> "SessionStats":
        cfg = cfg or load_config()
        sets = sorted({r.set_index for r in s.reps})
        reps_per_set = {k: 0 for k in sets}
        by_set: dict[int, dict[str, int]] = {k: {} for k in sets}
        totals: dict[str, int] = {}
        for r in s.reps:
            reps_per_set[r.set_index] += 1
            for code in r.present_faults(cfg):
                totals[code] = totals.get(code, 0) + 1
                by_set[r.set_index][code] = by_set[r.set_index].get(code, 0) + 1
        return cls(
            exercise=s.exercise,
            side=s.side,
            total=len(s.reps),
            sets=sets,
            reps_per_set=reps_per_set,
            fault_counts=dict(sorted(totals.items(), key=lambda t: (-t[1], t[0]))),
            fault_counts_by_set=by_set,
        )
