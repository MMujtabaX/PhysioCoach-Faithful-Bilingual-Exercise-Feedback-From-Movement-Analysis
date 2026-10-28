"""Real qwen2.5:3b outputs from the first LLM test run (2026-10-25), kept as regression cases.

Every one of them was correctly rejected. Each shows a different failure type
worth quoting in the report.
"""
import pytest

from physiocoach import RepRecord, check_rep, load_config

CFG = load_config()


def rec(side, faults, peak, knee, lean, dur):
    return RepRecord(exercise="side_leg_raise", side=side, set=1, rep=1, faults=faults,
                     features={"target_hip_abduction_deg": 40, "peak_hip_abduction_deg": peak,
                               "knee_angle_deg": knee, "trunk_lean_deg": lean, "rep_duration_s": dur})


CASES = [
    # (record, raw LLM text, expected issue fields)
    (rec("left", {"SHALLOW": .11, "KNEE_BENT": .57, "TORSO_LEAN": .07, "TOO_FAST": .2}, 40.6, 147.7, .7, 2.4),
     "Great effort, keep your knee straight. Your trunk is leaning a bit, try to keep it more upright.",
     {"hallucinated": ["TORSO_LEAN"], "missing": []}),
    (rec("left", {"SHALLOW": .79, "KNEE_BENT": .8, "TORSO_LEAN": .6, "TOO_FAST": .26}, 22, 157.6, 15.9, 2.3),
     "Raise your left leg a little more, you're 18 degrees below target. Keep your torso upright, don't lean.",
     {"hallucinated": [], "missing": ["KNEE_BENT"]}),  # omitted the most likely fault
    (rec("left", {"SHALLOW": .12, "KNEE_BENT": .77, "TORSO_LEAN": .88, "TOO_FAST": .82}, 42.3, 145.3, 27.1, .8),
     "Raise your left leg a little more, you're 15 degrees below target. Straighten your knee, it's bent. "
     "Your form was good, but you leaned too far to the left and moved too quickly. Try to maintain a more "
     "upright torso and keep your speed consistent.",
     {"hallucinated": ["SHALLOW"], "missing": [], "unexpected_numbers": [15]}),
    (rec("right", {"SHALLOW": .69, "KNEE_BENT": .97, "TORSO_LEAN": .24, "TOO_FAST": .36}, 20.8, 149.8, 1, 2.2),
     "Raise your right leg a little more, you're 19 degrees below target. Keep your knee straight, and keep "
     "your upper body upright.",
     {"hallucinated": ["TORSO_LEAN"], "missing": []}),
    (rec("right", {"SHALLOW": .35, "KNEE_BENT": .37, "TORSO_LEAN": .2, "TOO_FAST": .98}, 38.2, 167.7, 2.9, .7),
     "Great effort, but you're too fast. Keep your knee straight and your upper body upright.",
     {"hallucinated": ["KNEE_BENT", "TORSO_LEAN"], "missing": []}),
]


@pytest.mark.parametrize("record,text,expected", CASES)
def test_real_llm_outputs(record, text, expected):
    res = check_rep(text, record, CFG)
    assert not res.faithful
    for field, value in expected.items():
        assert getattr(res, field) == value, (field, res.issues)


@pytest.mark.parametrize("text", ["You're going too fast, slow it down.", "That one was too quick.",
                                  "You moved a bit fast there.", "Keep the knee straight.",
                                  "Your knee is bent, straighten it."])
def test_synonyms_are_recognised(text):
    r = rec("left", {"SHALLOW": .1, "KNEE_BENT": .9 if "knee" in text.lower() else .1,
                     "TORSO_LEAN": .1, "TOO_FAST": .1 if "knee" in text.lower() else .9}, 41, 150, 2, .8)
    assert check_rep(text, r, CFG).faithful, check_rep(text, r, CFG).issues
