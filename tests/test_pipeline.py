import random

from physiocoach import load_config
from physiocoach.numbers import UR_NUMBERS, extract_numbers_ur, normalize_digits, spell_digits, to_words_en
from physiocoach.pipeline import CoachPipeline
from physiocoach.records import make_rep, random_session
from physiocoach.translate import Translator, check_preservation

CFG = load_config()


class LyingGenerator:
    name = "liar"

    def rep(self, record):
        return "Keep your knee straight as you lift."  # claims a fault that is absent

    def session(self, session):
        return "Every rep was clean. Great session."


class FakeTranslator(Translator):
    name = "fake"

    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

    def _translate(self, text):
        return self.mapping(text)


def shallow_rep():
    return make_rep("side_leg_raise", {"SHALLOW"}, random.Random(4), side="left", cfg=CFG)


def test_hallucination_triggers_template_fallback():
    fb = CoachPipeline(LyingGenerator(), cfg=CFG).rep(shallow_rep())
    assert not fb.faithful_raw and fb.fallback
    assert "higher" in fb.text.lower() or "more" in fb.text.lower()


def test_ablation_without_check_keeps_raw_output():
    fb = CoachPipeline(LyingGenerator(), check=False, cfg=CFG).rep(shallow_rep())
    assert fb.text == "Keep your knee straight as you lift."
    assert not fb.fallback


def test_urdu_without_translator_uses_urdu_template():
    from physiocoach.generators import TemplateBackend

    fb = CoachPipeline(TemplateBackend(CFG), cfg=CFG).rep(shallow_rep(), lang="ur")
    assert fb.translation["method"] == "template"
    assert "بائیں" in fb.text


def test_good_translation_is_accepted():
    from physiocoach.generators import TemplateBackend

    rec = shallow_rep()
    deficit = str(int(round(40 - rec.features["peak_hip_abduction_deg"])))
    tr = FakeTranslator(lambda t: f"اپنی بائیں ٹانگ اوپر اٹھائیں، {deficit} ڈگری۔")
    fb = CoachPipeline(TemplateBackend(CFG), translator=tr, cfg=CFG).rep(rec, lang="ur")
    assert fb.translation["method"] == "fake" and not fb.translation["fallback"]


def test_translation_that_drops_a_number_is_rejected():
    from physiocoach.generators import TemplateBackend

    tr = FakeTranslator(lambda t: "اپنی بائیں ٹانگ اوپر اٹھائیں۔")
    fb = CoachPipeline(TemplateBackend(CFG), translator=tr, cfg=CFG).rep(shallow_rep(), lang="ur")
    assert fb.translation["fallback"] is True
    assert fb.translation["method"] == "template"


def test_session_pipeline_runs():
    from physiocoach.generators import TemplateBackend

    s = random_session(random.Random(3), "shoulder_abduction", n_sets=2, cfg=CFG)
    fb = CoachPipeline(TemplateBackend(CFG), cfg=CFG).session(s, lang="ur")
    assert fb.faithful_raw and fb.text


# ------------------------------------------------------------------ numbers
def test_urdu_number_table_and_digits():
    assert len(UR_NUMBERS) == 101 and UR_NUMBERS[12] == "بارہ"
    assert normalize_digits("۱۲ ڈگری") == "12 ڈگری"
    assert extract_numbers_ur("۱۲ ڈگری اور بارہ") == [12, 12]
    assert spell_digits("about 12 degrees", "en") == "about twelve degrees"
    assert spell_digits("12 ڈگری", "ur") == "بارہ ڈگری"
    assert to_words_en(45) == "forty-five"


def test_preservation_rules():
    assert check_preservation("Lift your left leg, 12 degrees short.", "بائیں ٹانگ، ۱۲ ڈگری").ok
    assert not check_preservation("Lift your left leg, 12 degrees short.", "دائیں ٹانگ، 12 ڈگری").ok
    assert not check_preservation("Lift your left leg, 12 degrees short.", "بائیں ٹانگ، 21 ڈگری").ok
    assert check_preservation("That's exactly right, nice work.", "بالکل ٹھیک، بہت خوب۔").ok
