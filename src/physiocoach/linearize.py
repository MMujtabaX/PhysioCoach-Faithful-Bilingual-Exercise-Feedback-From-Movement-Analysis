"""Turns structured records into flat text for seq2seq models (Flan-T5, LLMs).

Data-to-text models read text, so a JSON record is "linearized" into a compact,
consistent string. The facts a cue may state (present faults and their deficits)
are spelled out explicitly, which makes it much easier for a small model to get
numbers right.

Example::

    task: rep cue | exercise: side leg raise | side: left | set 2 rep 7 |
    faults: shallow 0.91 (deficit 12 deg) | clear: knee bent, torso lean, too fast |
    peak hip abduction 28.0 | target hip abduction 40.0 | knee angle 172.0 | ...
"""

from __future__ import annotations

from .config import CoachConfig, load_config
from .facts import rep_facts, session_facts
from .schema import RepRecord, SessionRecord

_UNIT_SUFFIXES = {"_deg": "", "_s": "", "_mm": ""}


def _feature_name(name: str) -> str:
    for suffix in _UNIT_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ")


def linearize_rep(record: RepRecord, cfg: CoachConfig | None = None) -> str:
    cfg = cfg or load_config()
    facts = rep_facts(record, cfg)
    ex = facts.exercise
    parts = [
        "task: rep cue",
        f"exercise: {ex.display['en']}",
        f"side: {record.side or 'none'}",
        f"set {record.set_index} rep {record.rep}",
    ]
    if facts.present:
        items = []
        for code in facts.present:
            item = f"{ex.faults[code].label} {record.faults[code]:.2f}"
            vals = facts.values.get(code, {})
            if "deficit" in vals:
                item += f" (deficit {vals['deficit']} deg)"
            if "target_duration" in vals:
                item += f" (target {vals['target_duration']} s)"
            items.append(item)
        parts.append("faults: " + ", ".join(items))
    else:
        parts.append("faults: none")
    if facts.absent:
        parts.append("clear: " + ", ".join(ex.faults[c].label for c in facts.absent))
    for name, value in record.features.items():
        parts.append(f"{_feature_name(name)} {value:.1f}")
    return " | ".join(parts)


def linearize_session(session: SessionRecord, cfg: CoachConfig | None = None) -> str:
    cfg = cfg or load_config()
    facts = session_facts(session, cfg)
    ex = facts.exercise
    st = facts.stats
    parts = [
        "task: session summary",
        f"exercise: {ex.display['en']}",
        f"side: {session.side or 'none'}",
        f"reps {st.total}",
        f"sets {len(st.sets)}",
    ]
    for s in st.sets:
        counts = st.fault_counts_by_set[s]
        detail = ", ".join(f"{ex.faults[c].label} {n}" for c, n in sorted(counts.items(), key=lambda t: -t[1]))
        parts.append(f"set {s}: {st.reps_per_set[s]} reps" + (f", {detail}" if detail else ", clean"))
    if facts.occurred:
        parts.append("totals: " + ", ".join(f"{ex.faults[c].label} {st.fault_counts[c]}" for c in facts.occurred))
        parts.append(f"main issue: {ex.faults[facts.top].label}")
    else:
        parts.append("totals: none")
    if facts.fatigue:
        parts.append(f"fatigue: {ex.faults[facts.fatigue.fault].label} rises to {facts.fatigue.last_count} in set {facts.fatigue.last_set}")
    return " | ".join(parts)
