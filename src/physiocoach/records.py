"""Synthetic record generator.

Produces realistic rep and session records so the NLP system can be built,
trained and graded without any computer-vision code. Feature values are sampled
so they agree with the fault labels (a SHALLOW rep really is short of target).
"""

from __future__ import annotations

import random
from typing import Iterable

from .config import CoachConfig, ExerciseSpec, load_config
from .schema import RepRecord, SessionRecord


def _sample_features(ex: ExerciseSpec, present: set[str], rng: random.Random) -> dict[str, float]:
    values: dict[str, float] = {}
    # fixed values first so relative features can use them
    for name, f in ex.features.items():
        if f.fixed is not None:
            values[name] = float(f.fixed)
    for name, f in ex.features.items():
        if f.fixed is not None:
            continue
        rng_range = f.normal
        for code, r in f.faults.items():
            if code in present:
                rng_range = r
                break
        if rng_range is None:
            raise ValueError(f"feature {name} has no range")
        v = rng.uniform(*rng_range)
        if f.relative_to:
            v += values[f.relative_to]
        values[name] = round(v, 1)
    return values


def make_rep(
    exercise: str,
    present: Iterable[str],
    rng: random.Random,
    side: str | None = None,
    set_index: int = 1,
    rep: int = 1,
    cfg: CoachConfig | None = None,
) -> RepRecord:
    """Build one rep whose fault probabilities and features match ``present``."""
    cfg = cfg or load_config()
    ex = cfg.exercise(exercise)
    present = set(present)
    unknown = present - set(ex.faults)
    if unknown:
        raise ValueError(f"unknown faults {unknown} for {exercise}")
    th = cfg.fault_threshold
    faults = {}
    for code in ex.faults:
        if code in present:
            p = rng.uniform(th + 0.05, 0.99)
        else:
            p = rng.uniform(0.01, th - 0.1)
        faults[code] = round(p, 2)
    if ex.sided and side is None:
        side = rng.choice(["left", "right"])
    return RepRecord(
        exercise=exercise,
        side=side if ex.sided else None,
        set=set_index,
        rep=rep,
        faults=faults,
        features=_sample_features(ex, present, rng),
    )


def random_rep(
    rng: random.Random,
    exercise: str | None = None,
    p_fault: float = 0.3,
    cfg: CoachConfig | None = None,
) -> RepRecord:
    """A random rep: each fault is independently present with ``p_fault``."""
    cfg = cfg or load_config()
    exercise = exercise or rng.choice(sorted(cfg.exercises))
    ex = cfg.exercise(exercise)
    present = {c for c in ex.faults if rng.random() < p_fault}
    return make_rep(
        exercise,
        present,
        rng,
        set_index=rng.randint(1, 3),
        rep=rng.randint(1, 15),
        cfg=cfg,
    )


def random_session(
    rng: random.Random,
    exercise: str | None = None,
    n_sets: int | None = None,
    reps_per_set: int | None = None,
    cfg: CoachConfig | None = None,
) -> SessionRecord:
    """A session with 0-2 recurring faults whose frequency can rise with fatigue."""
    cfg = cfg or load_config()
    exercise = exercise or rng.choice(sorted(cfg.exercises))
    ex = cfg.exercise(exercise)
    n_sets = n_sets or rng.randint(1, 3)
    side = rng.choice(["left", "right"]) if ex.sided else None

    codes = list(ex.faults)
    n_problems = rng.choices([0, 1, 2], weights=[0.2, 0.5, 0.3])[0]
    problems = rng.sample(codes, n_problems)
    base = {c: rng.uniform(0.1, 0.4) for c in problems}
    fatigue = {c: rng.choice([0.0, 0.0, 0.2, 0.35]) for c in problems}

    reps: list[RepRecord] = []
    for s in range(1, n_sets + 1):
        n_reps = reps_per_set or rng.randint(8, 15)
        for r in range(1, n_reps + 1):
            present = set()
            for c in codes:
                p = base.get(c, 0.02) + fatigue.get(c, 0.0) * (s - 1)
                if rng.random() < min(p, 0.95):
                    present.add(c)
            reps.append(make_rep(exercise, present, rng, side=side, set_index=s, rep=r, cfg=cfg))
    return SessionRecord(exercise=exercise, side=side, reps=reps)
