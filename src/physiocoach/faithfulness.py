"""Faithfulness checker: does a piece of feedback say only what the input supports?

For a rep cue, the text must

* correct every *required* fault (the top present faults),
* correct no fault that is absent (a hallucinated correction),
* praise no fault that is present,
* mention only numbers that come from the record (deficits, rep/set numbers),
* mention the correct side (or no side), never the opposite one.

For a session summary, the text must mention every notable fault, mention no
fault that never occurred, and use only numbers derivable from the session.

Detection uses the regexes in ``config/exercises.yaml``, so it works on output
from any generator (templates, Flan-T5, an LLM) as long as it uses natural
coaching vocabulary. Unusual phrasings can slip through; that limitation is
measured, not hidden, in the evaluation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .config import CoachConfig, load_config
from .facts import rep_facts, session_facts
from .numbers import NUMBER_TOKEN, extract_numbers_en, parse_number_en
from .schema import RepRecord, SessionRecord

_LEFT = re.compile(r"\bleft\b", re.IGNORECASE)
_RIGHT = re.compile(r"\bright\b(?! (there|now|away|on))", re.IGNORECASE)
# "That's exactly right" is praise, not a side
_RIGHT_IDIOM = re.compile(r"(exactly|just|that's|thats|all) right\b", re.IGNORECASE)


@dataclass
class CheckResult:
    faithful: bool
    mentioned: list[str] = field(default_factory=list)
    praised: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    hallucinated: list[str] = field(default_factory=list)
    wrong_praise: list[str] = field(default_factory=list)
    unexpected_numbers: list[int] = field(default_factory=list)
    wrong_claims: list[str] = field(default_factory=list)   # a number attached to the wrong fact
    misdescribed: list[str] = field(default_factory=list)   # a fault described in the wrong direction
    side_error: bool = False

    @property
    def issues(self) -> list[str]:
        out = []
        if self.missing:
            out.append(f"missing: {', '.join(self.missing)}")
        if self.hallucinated:
            out.append(f"hallucinated: {', '.join(self.hallucinated)}")
        if self.wrong_praise:
            out.append(f"praised a present fault: {', '.join(self.wrong_praise)}")
        if self.unexpected_numbers:
            out.append(f"unsupported numbers: {self.unexpected_numbers}")
        out.extend(self.wrong_claims)
        out.extend(self.misdescribed)
        if self.side_error:
            out.append("wrong side mentioned")
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["issues"] = self.issues
        return d


def _matches(patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _misdescribed(text: str, ex) -> list[str]:
    """Wording that misdescribes a fault of this exercise (e.g. 'lean forward' for a sideways torso lean)."""
    problems = []
    for f in ex.faults.values():
        for p in f.contradict:
            m = p.search(text)
            if m:
                problems.append(f"describes {f.label} wrongly ('{m.group(0)}')")
    return problems


def _side_error(text: str, side: str | None) -> bool:
    cleaned = _RIGHT_IDIOM.sub(" ", text)
    says_left = bool(_LEFT.search(cleaned))
    says_right = bool(_RIGHT.search(cleaned))
    if side is None:
        return says_left or says_right
    if side == "left":
        return says_right
    return says_left


# Claims that tie a number to a specific fact. Each regex captures the number token.
_DEGREES = re.compile(NUMBER_TOKEN + r"\s*(?:more\s+)?(?:degrees?|deg|°)", re.IGNORECASE)
_SECONDS = re.compile(NUMBER_TOKEN + r"\s*(?:seconds?|secs?|s\b)", re.IGNORECASE)
_TOTAL = re.compile(NUMBER_TOKEN + r"\s+reps?\s+(?:across|over|in)\s+" + NUMBER_TOKEN + r"\s+sets?", re.IGNORECASE)
_SETS_ONLY = re.compile(r"\b(?:across|over|in)\s+" + NUMBER_TOKEN + r"\s+sets?\b", re.IGNORECASE)
_COUNT = re.compile(r"(?:^|[.!?;,]\s*|\band\s+)" + NUMBER_TOKEN + r"\s+(?:of\s+(?:the\s+|your\s+)?)?reps?\s+"
                    r"(?:were|was|had|showed|came|looked)\b([^.!?;]*)", re.IGNORECASE)
_IN_SET = re.compile(r"\bset\s+" + NUMBER_TOKEN, re.IGNORECASE)


def _rep_claims(text: str, facts) -> list[str]:
    """Numbers attached to the wrong fact in a rep cue (e.g. 'about 3 degrees' when 12 are missing)."""
    deficits = {v["deficit"] for v in facts.values.values() if "deficit" in v}
    durations = {v["target_duration"] for v in facts.values.values() if "target_duration" in v}
    problems = []
    for m in _DEGREES.finditer(text):
        n = parse_number_en(m.group(1))
        if n is not None and n not in deficits:
            problems.append(f"says {n} degrees, but the analysis gives {', '.join(map(str, sorted(deficits))) or 'no degree value'}")
    for m in _SECONDS.finditer(text):
        n = parse_number_en(m.group(1))
        if n is not None and n not in durations:
            problems.append(f"says {n} seconds, but the target is {', '.join(map(str, sorted(durations))) or 'not given'}")
    return problems


def _session_claims(text: str, facts) -> list[str]:
    """Counts attached to the wrong fact in a session summary."""
    ex, st = facts.exercise, facts.stats
    problems = []
    for m in _TOTAL.finditer(text):
        total, sets = parse_number_en(m.group(1)), parse_number_en(m.group(2))
        if total is not None and total != st.total:
            problems.append(f"says {total} reps in total, but there were {st.total}")
        if sets is not None and sets != len(st.sets):
            problems.append(f"says {sets} sets, but there were {len(st.sets)}")
    if not _TOTAL.search(text):
        for m in _SETS_ONLY.finditer(text):
            sets = parse_number_en(m.group(1))
            if sets is not None and sets != len(st.sets):
                problems.append(f"says {sets} sets, but there were {len(st.sets)}")
    for m in _COUNT.finditer(text):
        n, phrase = parse_number_en(m.group(1)), m.group(2).lower()
        if n is None:
            continue
        codes = [c for c, f in ex.faults.items() if _matches(f.summary_detect, phrase) or _matches(f.detect, phrase)]
        for c in codes:
            actual = st.fault_counts.get(c, 0)
            if n != actual:
                problems.append(f"says {n} {'rep was' if n == 1 else 'reps were'} {ex.faults[c].label}, "
                                f"but it was {actual}")
    if facts.fatigue:
        for m in _IN_SET.finditer(text):
            n = parse_number_en(m.group(1))
            if n is not None and n not in st.sets:
                problems.append(f"mentions set {n}, but the session had sets {', '.join(map(str, st.sets))}")
    return problems


def check_rep(text: str, record: RepRecord, cfg: CoachConfig | None = None) -> CheckResult:
    cfg = cfg or load_config()
    facts = rep_facts(record, cfg)
    ex = facts.exercise
    low = text.lower()

    mentioned = [c for c, f in ex.faults.items() if _matches(f.detect, low)]
    praised = [c for c, f in ex.faults.items() if _matches(f.praise_detect, low) and c not in mentioned]
    nums = extract_numbers_en(text)

    res = CheckResult(
        faithful=True,
        mentioned=mentioned,
        praised=praised,
        missing=[c for c in facts.required if c not in mentioned],
        hallucinated=[c for c in mentioned if c not in facts.present],
        wrong_praise=[c for c in praised if c in facts.present],
        unexpected_numbers=sorted({n for n in nums if n not in facts.allowed_numbers}),
        wrong_claims=_rep_claims(text, facts),
        misdescribed=_misdescribed(text, ex),
        side_error=_side_error(text, facts.side),
    )
    res.faithful = not res.issues and bool(text.strip())
    return res


def check_session(text: str, session: SessionRecord, cfg: CoachConfig | None = None) -> CheckResult:
    cfg = cfg or load_config()
    facts = session_facts(session, cfg)
    ex = facts.exercise
    low = text.lower()

    mentioned = [c for c, f in ex.faults.items() if _matches(f.summary_detect, low) or _matches(f.detect, low)]
    nums = extract_numbers_en(text)
    res = CheckResult(
        faithful=True,
        mentioned=mentioned,
        missing=[c for c in facts.notable if c not in mentioned],
        hallucinated=[c for c in mentioned if c not in facts.occurred],
        unexpected_numbers=sorted({n for n in nums if n not in facts.allowed_numbers}),
        wrong_claims=_session_claims(text, facts),
        misdescribed=_misdescribed(text, ex),
        side_error=_side_error(text, session.side) if session.side is None else False,
    )
    res.faithful = not res.issues and bool(text.strip())
    return res


# ----------------------------------------------------------------- annotation
def _merge(problems: list[str], claims: list[str]) -> list[str]:
    """Drop the generic 'number N is not supported' line when a claim already explains N."""
    explained = {w for c in claims for w in re.findall(r"\bsays (\d+)", c)}
    kept = [p for p in problems if not any(f"number {n} " in p for n in explained)]
    return kept + claims
def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def annotate_rep(text: str, record: RepRecord, cfg: CoachConfig | None = None) -> dict:
    """Per-sentence view of a rep cue: what each sentence says and what is wrong with it.

    Used by the comparison page to highlight the exact sentence that hallucinates.
    """
    cfg = cfg or load_config()
    facts = rep_facts(record, cfg)
    ex = facts.exercise
    out, mentioned_all = [], set()
    for sentence in _sentences(text):
        low = sentence.lower()
        mentioned = [c for c, f in ex.faults.items() if _matches(f.detect, low)]
        praised = [c for c, f in ex.faults.items() if _matches(f.praise_detect, low) and c not in mentioned]
        mentioned_all.update(mentioned)
        problems = []
        for c in mentioned:
            if c not in facts.present:
                problems.append(f"corrects “{ex.faults[c].label}”, which was not detected")
        for c in praised:
            if c in facts.present:
                problems.append(f"praises “{ex.faults[c].label}”, which was a fault")
        for n in sorted({n for n in extract_numbers_en(sentence) if n not in facts.allowed_numbers}):
            problems.append(f"the number {n} is not supported by the input")
        if _side_error(sentence, facts.side):
            problems.append("mentions the wrong side")
        problems = _merge(problems, _rep_claims(sentence, facts))
        problems += _misdescribed(sentence, ex)
        out.append({
            "text": sentence,
            "mentions": [ex.faults[c].label for c in mentioned],
            "praises": [ex.faults[c].label for c in praised],
            "problems": problems,
        })
    missed = [ex.faults[c].label for c in facts.required if c not in mentioned_all]
    return {"sentences": out, "missed": missed}


def annotate_session(text: str, session: SessionRecord, cfg: CoachConfig | None = None) -> dict:
    cfg = cfg or load_config()
    facts = session_facts(session, cfg)
    ex = facts.exercise
    out, mentioned_all = [], set()
    for sentence in _sentences(text):
        low = sentence.lower()
        mentioned = [c for c, f in ex.faults.items() if _matches(f.summary_detect, low) or _matches(f.detect, low)]
        mentioned_all.update(mentioned)
        problems = [f"mentions “{ex.faults[c].label}”, which never happened" for c in mentioned
                    if c not in facts.occurred]
        problems += [f"the number {n} is not supported by the session"
                     for n in sorted({n for n in extract_numbers_en(sentence) if n not in facts.allowed_numbers})]
        problems = _merge(problems, _session_claims(sentence, facts))
        problems += _misdescribed(sentence, ex)
        out.append({"text": sentence, "mentions": [ex.faults[c].label for c in mentioned], "praises": [],
                    "problems": problems})
    missed = [ex.faults[c].label for c in facts.notable if c not in mentioned_all]
    return {"sentences": out, "missed": missed}
