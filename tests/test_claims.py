"""Numbers must match the fact they describe, not just appear somewhere in the input."""
import random

from physiocoach import RepRecord, SessionRecord, check_rep, check_session, load_config
from physiocoach.numbers import parse_number_en
from physiocoach.records import make_rep

CFG = load_config()


def session(n_sets=2, per_set=10, knee_bent_reps=(1, 3, 5, 7, 12, 14, 16), shallow_reps=(2, 9, 18)):
    rng, reps, i = random.Random(0), [], 0
    for s in range(1, n_sets + 1):
        for r in range(1, per_set + 1):
            i += 1
            faults = set()
            if i in knee_bent_reps:
                faults.add("KNEE_BENT")
            if i in shallow_reps:
                faults.add("SHALLOW")
            reps.append(make_rep("side_leg_raise", faults, rng, side="right", set_index=s, rep=r, cfg=CFG))
    return SessionRecord(exercise="side_leg_raise", side="right", reps=reps)


def test_tiny_model_mistake_is_caught():
    # 20 reps, 2 sets, knee bent 7, shallow 3: every number below exists *somewhere* in the input
    s = session()
    text = ("You completed 20 reps across 3 sets. 2 reps were done with a bent knee. 3 reps were too shallow. "
            "Next time, keep the knee locked straight.")
    res = check_session(text, s, CFG)
    assert not res.faithful
    assert "says 3 sets, but there were 2" in res.wrong_claims
    assert "says 2 reps were knee bent, but it was 7" in res.wrong_claims
    assert not any("shallow" in c for c in res.wrong_claims)


def test_correct_summary_passes():
    s = session()
    text = ("You completed 20 reps across 2 sets. 7 reps were done with a bent knee. 3 reps were too shallow. "
            "Next time, keep the knee locked straight.")
    assert check_session(text, s, CFG).faithful, check_session(text, s, CFG).issues


def test_degrees_must_match_the_deficit():
    rec = RepRecord(exercise="side_leg_raise", side="left", set=1, rep=3,
                    faults={"SHALLOW": .9, "KNEE_BENT": .1, "TORSO_LEAN": .1, "TOO_FAST": .1},
                    features={"target_hip_abduction_deg": 40, "peak_hip_abduction_deg": 28})
    assert check_rep("Lift your left leg higher, about 12 degrees short.", rec, CFG).faithful
    bad = check_rep("Lift your left leg higher, about 3 degrees short.", rec, CFG)  # 3 = the rep number
    assert not bad.faithful and "says 3 degrees" in bad.wrong_claims[0]
    words = check_rep("Lift your left leg higher, about three degrees short.", rec, CFG)
    assert not words.faithful


def test_number_words():
    assert parse_number_en("12") == 12
    assert parse_number_en("twelve") == 12
    assert parse_number_en("twenty-one") == 21
    assert parse_number_en("twenty one") == 21
    assert parse_number_en("about twelve") == 12
    assert parse_number_en("few") is None
