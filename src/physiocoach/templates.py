"""Template-based feedback (Method A).

Plays three roles in the project:

1. the baseline in the model comparison,
2. the generator of reference texts for the training dataset,
3. the always-faithful *fallback* when a model's output fails the checker
   or a translation fails the preservation check.

A template output is built in two steps: :meth:`plan_rep` decides *what* to
say (which faults, which wording variant), :meth:`render_rep` says it in a
language. Rendering the same plan in English and Urdu gives two texts with
identical content, which is what the Urdu fallback relies on.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from .config import CoachConfig, load_config
from .facts import RepFacts, SessionFacts, rep_facts, session_facts
from .schema import RepRecord, SessionRecord

LANGS = ("en", "ur")


@dataclass
class RepPlan:
    facts: RepFacts
    corrections: list[tuple[str, int]] = field(default_factory=list)  # (fault, variant index)
    praise: tuple[str, int] | None = None
    clean_variant: int = 0


@dataclass
class SessionPlan:
    facts: SessionFacts
    opening: int = 0
    fatigue: int = 0
    clean: int = 0
    rest_good: bool = False


def _stable_rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _pick(rng: random.Random, options: list[str]) -> int:
    return rng.randrange(len(options)) if options else 0


def _choose(options: list[str], i: int) -> str:
    return options[i % len(options)]


class TemplateGenerator:
    name = "template"

    def __init__(self, cfg: CoachConfig | None = None, seed: int = 0, praise_prob: float = 0.6):
        self.cfg = cfg or load_config()
        self.seed = seed
        self.praise_prob = praise_prob

    # ------------------------------------------------------------------ rep
    def plan_rep(self, record: RepRecord, rng: random.Random | None = None) -> RepPlan:
        rng = rng or _stable_rng(self.seed, record.model_dump_json())
        facts = rep_facts(record, self.cfg)
        ex = facts.exercise
        plan = RepPlan(facts=facts)
        for code in facts.required:
            plan.corrections.append((code, _pick(rng, ex.faults[code].cue["en"])))
        if len(plan.corrections) <= 1 and facts.absent and rng.random() < self.praise_prob:
            code = rng.choice(facts.absent)
            if ex.faults[code].praise.get("en"):
                plan.praise = (code, _pick(rng, ex.faults[code].praise["en"]))
        plan.clean_variant = _pick(rng, self.cfg.clean_rep["en"])
        return plan

    def _fill(self, text: str, facts: RepFacts, code: str | None, lang: str) -> str:
        side = self.cfg.sides.get(facts.side or "", {"en": "", "ur": ""})
        values = dict(facts.values.get(code, {})) if code else {}
        return text.format(
            side=side["en"],
            side_ur=side["ur"],
            limb=facts.exercise.limb["en"],
            limb_ur=facts.exercise.limb["ur"],
            **values,
        )

    def render_rep(self, plan: RepPlan, lang: str = "en") -> str:
        if lang not in LANGS:
            raise ValueError(f"lang must be one of {LANGS}")
        ex = plan.facts.exercise
        parts: list[str] = []
        if not plan.corrections:
            parts.append(_choose(self.cfg.clean_rep[lang], plan.clean_variant))
        for code, i in plan.corrections:
            parts.append(self._fill(_choose(ex.faults[code].cue[lang], i), plan.facts, code, lang))
        if plan.praise:
            code, i = plan.praise
            options = ex.faults[code].praise.get(lang) or ex.faults[code].praise["en"]
            parts.append(self._fill(_choose(options, i), plan.facts, None, lang))
        text = " ".join(parts)
        return text[:1].upper() + text[1:] if lang == "en" else text

    def rep(self, record: RepRecord, lang: str = "en", rng: random.Random | None = None) -> str:
        return self.render_rep(self.plan_rep(record, rng), lang)

    # -------------------------------------------------------------- session
    def plan_session(self, session: SessionRecord, rng: random.Random | None = None) -> SessionPlan:
        rng = rng or _stable_rng(self.seed, session.model_dump_json())
        st = self.cfg.session_text
        facts = session_facts(session, self.cfg)
        return SessionPlan(
            facts=facts,
            opening=_pick(rng, st["opening"]["en"]),
            fatigue=_pick(rng, st["fatigue"]["en"]),
            clean=_pick(rng, st["clean"]["en"]),
            rest_good=bool(facts.occurred) and len(facts.occurred) < len(facts.exercise.faults) and rng.random() < 0.5,
        )

    def render_session(self, plan: SessionPlan, lang: str = "en") -> str:
        st = self.cfg.session_text
        f = plan.facts
        ex = f.exercise
        n_sets = len(f.stats.sets)
        sets_phrase = st["sets_phrase_en"]["one"] if n_sets == 1 else st["sets_phrase_en"]["many"].format(sets=n_sets)
        out = [_choose(st["opening"][lang], plan.opening).format(total=f.stats.total, sets=n_sets, sets_phrase=sets_phrase)]

        if not f.occurred:
            out.append(_choose(st["clean"][lang], plan.clean))
            return " ".join(out)

        for code in f.occurred:
            count = f.stats.fault_counts[code]
            phrase = ex.faults[code].summary[lang]
            key = f"{lang}_one" if count == 1 else f"{lang}_many"
            line = st["fault_line"][key].format(count=count, phrase=phrase)
            out.append(line)
            if f.fatigue and code == f.fatigue.fault:
                out.append(
                    _choose(st["fatigue"][lang], plan.fatigue).format(
                        last_count=f.fatigue.last_count, last_set=f.fatigue.last_set
                    )
                )
        if plan.rest_good:
            out.append(st["rest_good"][lang][0])
        top = ex.faults[f.top]
        tip = top.tip[lang].format(**{k: int(v) for k, v in top.values.items()})
        out.append(_choose(st["tip"][lang], 0).format(tip=tip))
        return " ".join(out)

    def session(self, session: SessionRecord, lang: str = "en", rng: random.Random | None = None) -> str:
        return self.render_session(self.plan_session(session, rng), lang)


def all_rep_sentences(cfg: CoachConfig | None = None, lang: str = "en", max_deficit: int = 45) -> list[str]:
    """Every sentence a live rep cue can contain, for pre-generating audio.

    Cues are built from one sentence per correction or praise, so caching each
    of these makes any cue play instantly (see scripts/precompute_audio.py).
    Deficits cover 1..max_deficit degrees, so real-world values (not just the
    synthetic ranges) are covered too.
    """
    cfg = cfg or load_config()
    out: dict[str, None] = {}
    for s in cfg.clean_rep[lang]:
        out[s] = None
    for ex in cfg.exercises.values():
        sides = ["left", "right"] if ex.sided else [None]
        for code, fault in ex.faults.items():
            for template in fault.cue[lang]:
                deficits = range(1, max_deficit + 1) if "{deficit}" in template else [None]
                for side in sides:
                    side_words = cfg.sides.get(side or "", {"en": "", "ur": ""})
                    for d in deficits:
                        values = {k: int(v) for k, v in fault.values.items()}
                        if d is not None:
                            values["deficit"] = d
                        text = template.format(side=side_words["en"], side_ur=side_words["ur"],
                                               limb=ex.limb["en"], limb_ur=ex.limb["ur"], **values)
                        out[text[:1].upper() + text[1:] if lang == "en" else text] = None
            for s in fault.praise.get(lang) or []:
                out[s] = None
    return list(out)
