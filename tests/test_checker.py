import pytest

from physiocoach import RepRecord, check_rep, load_config
from physiocoach.faithfulness import annotate_rep

CFG = load_config()


def rec(faults: dict, side="left", exercise="side_leg_raise", peak=28.0):
    base = {"SHALLOW": 0.1, "KNEE_BENT": 0.1, "TORSO_LEAN": 0.1, "TOO_FAST": 0.1}
    base.update(faults)
    return RepRecord(
        exercise=exercise, side=side, set=1, rep=3, faults=base,
        features={"target_hip_abduction_deg": 40.0, "peak_hip_abduction_deg": peak, "knee_angle_deg": 172.0,
                  "trunk_lean_deg": 4.0, "rep_duration_s": 2.0},
    )


def test_correct_cue_passes():
    r = check_rep("Lift your left leg higher, about 12 degrees short.", rec({"SHALLOW": 0.9}), CFG)
    assert r.faithful, r.issues


def test_hallucinated_fault_is_caught():
    r = check_rep("Lift your left leg higher, about 12 degrees short. Keep your knee straight as you lift.",
                  rec({"SHALLOW": 0.9}), CFG)
    assert not r.faithful
    assert r.hallucinated == ["KNEE_BENT"]


def test_missing_fault_is_caught():
    r = check_rep("Great rep, keep it up.", rec({"SHALLOW": 0.9}), CFG)
    assert r.missing == ["SHALLOW"]


def test_praising_a_present_fault_is_caught():
    r = check_rep("Lift your left leg higher, about 12 degrees short. Knee is nice and straight.",
                  rec({"SHALLOW": 0.9, "KNEE_BENT": 0.7}), CFG)
    assert not r.faithful
    assert "KNEE_BENT" in r.missing or "KNEE_BENT" in r.wrong_praise


def test_wrong_number_is_caught():
    r = check_rep("Lift your left leg higher, about 20 degrees short.", rec({"SHALLOW": 0.9}), CFG)
    assert r.unexpected_numbers == [20]


def test_number_words_are_checked_too():
    r = check_rep("Lift your left leg higher, about twenty degrees short.", rec({"SHALLOW": 0.9}), CFG)
    assert 20 in r.unexpected_numbers


def test_wrong_side_is_caught():
    r = check_rep("Lift your right leg higher, about 12 degrees short.", rec({"SHALLOW": 0.9}), CFG)
    assert r.side_error


@pytest.mark.parametrize("text", ["That's exactly right, nice work.", "Great rep, keep it up."])
def test_idiomatic_right_is_not_a_side(text):
    assert check_rep(text, rec({}), CFG).faithful


def test_only_top_two_faults_are_required():
    r = check_rep("Lift your left leg higher, about 12 degrees short. Keep your knee straight as you lift.",
                  rec({"SHALLOW": 0.95, "KNEE_BENT": 0.9, "TORSO_LEAN": 0.6}), CFG)
    assert r.faithful, r.issues


def squat(faults: dict, peak=101.0):
    base = {"SHALLOW": 0.1, "KNEES_CAVE": 0.1, "FORWARD_LEAN": 0.1, "TOO_FAST": 0.1}
    base.update(faults)
    return RepRecord(
        exercise="squat", side=None, set=1, rep=3, faults=base,
        features={"target_knee_flexion_deg": 100.0, "peak_knee_flexion_deg": peak, "knee_valgus_deg": 3.0,
                  "trunk_flexion_deg": 60.0, "rep_duration_s": 2.0},
    )


def shoulder(faults: dict, peak=76.0):
    base = {"SHALLOW": 0.1, "SHRUG": 0.1, "ELBOW_BENT": 0.1, "TOO_FAST": 0.1}
    base.update(faults)
    return RepRecord(
        exercise="shoulder_abduction", side="right", set=1, rep=3, faults=base,
        features={"target_shoulder_abduction_deg": 90.0, "peak_shoulder_abduction_deg": peak,
                  "shoulder_elevation_deg": 2.0, "elbow_angle_deg": 172.0, "rep_duration_s": 2.0},
    )


# each SHALLOW record is 14 degrees short
SHALLOW_CASES = {
    "side_leg_raise": (lambda: rec({"SHALLOW": 0.9}, side="right", peak=26.0),
                       "Go lower, you're 14 degrees short of depth.",
                       "Lift your right leg higher, about 14 degrees short."),
    "shoulder_abduction": (lambda: shoulder({"SHALLOW": 0.9}),
                           "Raise your right leg higher, about 14 degrees to go.",
                           "Raise your right arm higher, about 14 degrees to go."),
    "squat": (lambda: squat({"SHALLOW": 0.9}, peak=86.0),
              "Lift your leg higher, about 14 degrees short.",
              "Go lower, you're 14 degrees short of depth."),
}


@pytest.mark.parametrize("exercise", SHALLOW_CASES)
def test_shallow_in_another_exercises_words_is_caught(exercise):
    make, bad, _ = SHALLOW_CASES[exercise]
    r = check_rep(bad, make(), CFG)
    assert not r.faithful
    assert any(m.startswith("describes shallow wrongly") for m in r.misdescribed), r.issues


@pytest.mark.parametrize("make, text", [
    (lambda: rec({}), "Chest stayed up nicely."),
    (lambda: rec({"TORSO_LEAN": 0.9}), "Keep your shoulder down, don't lean sideways."),
    (lambda: shoulder({}), "Knee is nice and straight."),
    (lambda: shoulder({"SHRUG": 0.9}), "Spread your shoulder wider, over your toes."),
    (lambda: shoulder({"SHALLOW": 0.9}, peak=68.0),
     "Raise your right arm a bit more, aiming for 22 degrees above shoulder height."),
])
def test_cross_exercise_wording_found_by_model_sweep_is_caught(make, text):
    r = check_rep(text, make(), CFG)
    assert not r.faithful
    assert r.misdescribed, r.issues


@pytest.mark.parametrize("exercise", SHALLOW_CASES)
def test_shallow_in_own_words_passes(exercise):
    make, _, good = SHALLOW_CASES[exercise]
    r = check_rep(good, make(), CFG)
    assert r.faithful, r.issues


def test_sideways_lean_described_as_forward_is_caught():
    r = check_rep("Keep your upper body upright, don't lean forward.", rec({"TORSO_LEAN": 0.9}), CFG)
    assert not r.faithful
    assert r.misdescribed == ["describes torso lean wrongly ('lean forward')"]


def test_sideways_lean_described_correctly_passes():
    r = check_rep("Keep your upper body upright, don't lean sideways.", rec({"TORSO_LEAN": 0.9}), CFG)
    assert r.faithful, r.issues


def test_forward_lean_described_as_sideways_is_caught():
    r = check_rep("Don't lean sideways.", squat({"FORWARD_LEAN": 0.9}), CFG)
    assert not r.faithful
    assert r.misdescribed


def test_forward_lean_described_correctly_passes():
    r = check_rep("Keep your chest up.", squat({"FORWARD_LEAN": 0.9}), CFG)
    assert r.faithful, r.issues


def test_misdescription_is_flagged_on_its_sentence():
    a = annotate_rep("Keep your upper body upright. Don't lean forward.", rec({"TORSO_LEAN": 0.9}), CFG)
    assert a["sentences"][0]["problems"] == []
    assert a["sentences"][1]["problems"] == ["describes torso lean wrongly ('lean forward')"]
