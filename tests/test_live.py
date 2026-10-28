"""Live mode: instant Urdu and sentence-cached speech."""
import random
import wave

from physiocoach import RepRecord, TemplateGenerator, load_config
from physiocoach.generators import TemplateBackend
from physiocoach.pipeline import CoachPipeline
from physiocoach.records import random_rep
from physiocoach.templates import all_rep_sentences
from physiocoach.translate import Translator
from physiocoach.tts import Speaker, split_sentences

CFG = load_config()


class CountingSpeaker(Speaker):
    """Writes a short silent clip per sentence and counts real syntheses."""
    name = "fake"

    def __init__(self, cache_dir):
        super().__init__(cache_dir)
        self.calls = []

    def _synthesize(self, text, lang, path):
        self.calls.append(text)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x01\x00" * 1600)


class ExplodingTranslator(Translator):
    name = "boom"

    def _translate(self, text):
        raise AssertionError("live mode must not call the translator")


def test_every_live_cue_is_made_of_precomputable_sentences():
    gen = TemplateGenerator(CFG)
    known = {lang: set(all_rep_sentences(CFG, lang)) for lang in ("en", "ur")}
    rng = random.Random(0)
    for _ in range(2000):
        rec = random_rep(rng, p_fault=rng.choice([0.2, 0.5, 0.9]), cfg=CFG)
        for lang in ("en", "ur"):
            for sentence in split_sentences(gen.rep(rec, lang)):
                assert sentence in known[lang], (lang, sentence)


def test_live_mode_skips_translation(tmp_path):
    rec = random_rep(random.Random(1), "side_leg_raise", p_fault=0.9, cfg=CFG)
    pipe = CoachPipeline(TemplateBackend(CFG), translator=ExplodingTranslator(), cfg=CFG)
    fb = pipe.rep(rec, lang="ur", live=True)
    assert fb.translation["method"] == "template" and fb.text


def test_speech_reuses_cached_sentences(tmp_path):
    spk = CountingSpeaker(tmp_path)
    p1, cached1 = spk.speak("Keep your knee straight as you lift. Nice controlled pace.", "en")
    assert not cached1 and len(spk.calls) == 2
    p2, cached2 = spk.speak("Lift your left leg higher, about 12 degrees short. Nice controlled pace.", "en")
    assert len(spk.calls) == 3          # only the new sentence was synthesized
    p3, cached3 = spk.speak("Keep your knee straight as you lift.", "en")
    assert cached3 and len(spk.calls) == 3
    with wave.open(str(p2)) as wf:      # two clips joined with a pause
        assert wf.getnframes() > 2 * 1600


def test_api_rep_defaults_to_live():
    from fastapi.testclient import TestClient

    from physiocoach.api import create_app
    from physiocoach.factory import Registry, Settings

    client = TestClient(create_app(Registry(settings=Settings(translator="none", tts="none", warmup=False))))
    rec = RepRecord(**client.get("/api/sample/rep", params={"seed": 5, "p_fault": 0.8}).json())
    out = client.post("/api/feedback/rep", json={"record": rec.to_json_dict(), "lang": "ur"}).json()
    assert "live mode" in out["translation"]["reason"]
