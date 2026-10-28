"""What is *true* about a record.

Templates (what we say) and the faithfulness checker (what we are allowed to
say) both derive from these facts, so they always agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import CoachConfig, ExerciseSpec, FaultSpec, load_config
from .schema import RepRecord, SessionRecord, SessionStats


def fault_values(record: RepRecord, spec: FaultSpec) -> dict[str, int]:
    """Numbers a cue for this fault may mention (e.g. how many degrees short)."""
    values: dict[str, int] = {}
    if spec.deficit:
        target = record.features.get(spec.deficit["target"])
        actual = record.features.get(spec.deficit["value"])
        if target is not None and actual is not None:
            values["deficit"] = max(1, int(round(target - actual)))
    for k, v in spec.values.items():
        values[k] = int(v)
    return values


@dataclass
class RepFacts:
    exercise: ExerciseSpec
    side: str | None
    present: list[str]           # all present faults, highest probability first
    required: list[str]          # faults a cue must mention (top-N present)
    absent: list[str]
    values: dict[str, dict[str, int]] = field(default_factory=dict)
    allowed_numbers: set[int] = field(default_factory=set)


def rep_facts(record: RepRecord, cfg: CoachConfig | None = None) -> RepFacts:
    cfg = cfg or load_config()
    ex = cfg.exercise(record.exercise)
    present = record.present_faults(cfg)
    # Only faults the analysis actually measured can be praised or called "clear".
    # (PhysioLens, for example, may report only SHALLOW and KNEE_BENT.)
    absent = [c for c in ex.fault_codes if c in record.faults and c not in present]
    values = {c: fault_values(record, ex.faults[c]) for c in present}
    allowed = {record.rep, record.set_index}
    for v in values.values():
        allowed.update(v.values())
    return RepFacts(
        exercise=ex,
        side=record.side,
        present=present,
        required=present[: cfg.max_corrections],
        absent=absent,
        values=values,
        allowed_numbers=allowed,
    )


@dataclass
class Fatigue:
    fault: str
    last_set: int
    last_count: int


@dataclass
class SessionFacts:
    exercise: ExerciseSpec
    stats: SessionStats
    occurred: list[str]          # faults seen at least once, most frequent first
    notable: list[str]           # faults a summary must mention
    top: str | None
    fatigue: Fatigue | None
    allowed_numbers: set[int]


def session_facts(session: SessionRecord, cfg: CoachConfig | None = None) -> SessionFacts:
    cfg = cfg or load_config()
    ex = cfg.exercise(session.exercise)
    stats = SessionStats.from_session(session, cfg)
    occurred = list(stats.fault_counts)
    threshold = max(2, math.ceil(0.15 * stats.total))
    notable = [c for c in occurred if stats.fault_counts[c] >= threshold]
    top = occurred[0] if occurred else None

    fatigue = None
    if top and len(stats.sets) >= 2:
        first, last = stats.sets[0], stats.sets[-1]
        first_n = stats.fault_counts_by_set[first].get(top, 0)
        last_n = stats.fault_counts_by_set[last].get(top, 0)
        if last_n >= 3 and last_n >= 2 * first_n + 1:
            fatigue = Fatigue(fault=top, last_set=last, last_count=last_n)

    allowed = {stats.total, len(stats.sets), *stats.sets, *stats.fault_counts.values()}
    allowed.update(stats.reps_per_set.values())
    for counts in stats.fault_counts_by_set.values():
        allowed.update(counts.values())
    for c in occurred:
        allowed.update(int(v) for v in ex.faults[c].values.values())
    return SessionFacts(ex, stats, occurred, notable, top, fatigue, allowed)
