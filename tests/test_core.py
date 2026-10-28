import random

import pytest

from physiocoach import (
    RepRecord,
    TemplateGenerator,
    check_rep,
    check_session,
    linearize_rep,
    linearize_session,
    load_config,
)
from physiocoach.records import make_rep, random_rep, random_session

CFG = load_config()
GEN = TemplateGenerator(CFG)


@pytest.mark.parametrize("seed", range(5))
def test_template_reps_are_always_faithful(seed):
    rng = random.Random(seed)
    for _ in range(800):
        rec = random_rep(rng, p_fault=rng.choice([0.1, 0.4, 0.8]), cfg=CFG)
        text = GEN.rep(rec, rng=random.Random(rng.random()))
        res = check_rep(text, rec, CFG)
        assert res.faithful, (rec.present_faults(CFG), text, res.issues)


@pytest.mark.parametrize("seed", range(3))
def test_template_sessions_are_always_faithful(seed):
    rng = random.Random(seed)
    for _ in range(300):
        s = random_session(rng, cfg=CFG)
        text = GEN.session(s, rng=random.Random(rng.random()))
        res = check_session(text, s, CFG)
        assert res.faithful, (text, res.issues)


def test_features_agree_with_labels():
    rng = random.Random(0)
    rec = make_rep("side_leg_raise", {"SHALLOW"}, rng, side="left", cfg=CFG)
    assert rec.features["peak_hip_abduction_deg"] <= 40 - 6
    assert rec.faults["SHALLOW"] >= CFG.fault_threshold
    assert rec.faults["KNEE_BENT"] < CFG.fault_threshold
    clean = make_rep("side_leg_raise", set(), rng, side="left", cfg=CFG)
    assert clean.present_faults(CFG) == []


def test_urdu_render_has_same_content():
    rng = random.Random(1)
    rec = make_rep("side_leg_raise", {"SHALLOW"}, rng, side="left", cfg=CFG)
    plan = GEN.plan_rep(rec)
    en, ur = GEN.render_rep(plan, "en"), GEN.render_rep(plan, "ur")
    deficit = str(int(round(40 - rec.features["peak_hip_abduction_deg"])))
    assert "left" in en and deficit in en
    assert "بائیں" in ur and deficit in ur


def test_unsided_exercise_never_mentions_a_side():
    rng = random.Random(2)
    for _ in range(200):
        rec = random_rep(rng, "squat", p_fault=0.5, cfg=CFG)
        text = GEN.rep(rec).lower()
        assert " left" not in text and " right " not in text


def test_linearize_spells_out_deficit():
    rec = RepRecord(
        exercise="side_leg_raise", side="left", set=2, rep=7,
        faults={"SHALLOW": 0.91, "KNEE_BENT": 0.12, "TORSO_LEAN": 0.05, "TOO_FAST": 0.1},
        features={"target_hip_abduction_deg": 40, "peak_hip_abduction_deg": 28, "knee_angle_deg": 172,
                  "trunk_lean_deg": 4, "rep_duration_s": 2.1},
    )
    line = linearize_rep(rec, CFG)
    assert "shallow 0.91 (deficit 12 deg)" in line
    assert "clear: knee bent, torso lean, too fast" in line
    assert "side: left" in line


def test_linearize_session_reports_totals():
    s = random_session(random.Random(9), "squat", n_sets=2, cfg=CFG)
    line = linearize_session(s, CFG)
    assert line.startswith("task: session summary")
    assert "sets 2" in line


def test_unmeasured_faults_are_never_praised():
    """A PhysioLens-style record with only two measured faults."""
    for seed in range(300):
        rec = RepRecord(exercise="side_leg_raise", side="right", faults={"SHALLOW": 0.2, "KNEE_BENT": 0.1},
                        features={"target_hip_abduction_deg": 40, "peak_hip_abduction_deg": 41})
        text = TemplateGenerator(CFG, seed=seed).rep(rec)
        assert "torso" not in text.lower() and "pace" not in text.lower()
        assert check_rep(text, rec, CFG).faithful
    assert "clear: shallow, knee bent" in linearize_rep(rec, CFG)


def test_record_validation():
    with pytest.raises(ValueError):
        RepRecord(exercise="side_leg_raise", side="left", faults={"SHALLOW": 1.5})
    with pytest.raises(ValueError):
        RepRecord(exercise="side_leg_raise", faults={"SHALLOW": 0.9}).validate_against(CFG)  # side missing
    with pytest.raises(ValueError):
        RepRecord(exercise="side_leg_raise", side="left", faults={"WOBBLE": 0.9}).validate_against(CFG)
    with pytest.raises(ValueError):
        RepRecord(exercise="yoga", faults={}).validate_against(CFG)
