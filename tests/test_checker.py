import pytest

from physiocoach import RepRecord, check_rep, load_config

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
