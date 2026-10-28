"""Checker-guided decoding, tested with a fake model (no torch needed)."""
import random

from physiocoach import load_config
from physiocoach.generators import GuidedT5Backend
from physiocoach.pipeline import CoachPipeline
from physiocoach.records import make_rep

CFG = load_config()


class FakeGuided(GuidedT5Backend):
    def __init__(self, candidates, seed=0):  # skip model loading
        self.cfg, self.n_samples, self.top_p, self.temperature = CFG, len(candidates), 0.9, 1.0
        self.rng, self.last_stats, self._cands = random.Random(seed), {}, candidates
        self.fallback_calls = 0

    def _sample(self, prompt, max_new_tokens, n, top_p, temperature):
        return list(self._cands)

    def _generate(self, prompt, max_new_tokens):
        self.fallback_calls += 1
        return self._cands[-1]


def rec():
    r = make_rep("side_leg_raise", {"KNEE_BENT"}, random.Random(0), side="left", cfg=CFG)
    return r


def test_unfaithful_candidates_are_never_chosen():
    good = ["Keep your knee straight as you lift.", "Straighten your knee, it's bending."]
    bad = ["Lift your left leg higher, about 12 degrees short.", "Great rep, keep it up."]
    picks = set()
    for seed in range(40):
        g = FakeGuided(bad + good + bad[:1] + [good[0]], seed=seed)
        out = g.rep(rec())
        assert out in good
        picks.add(out)
        assert g.last_stats == {"candidates": 6, "distinct": 4, "faithful": 2, "fallback_decode": False}
        assert g.fallback_calls == 0    # no extra decode when a sample passes
    assert picks == set(good)  # variety: both faithful phrasings get used


def test_no_faithful_candidate_falls_back_to_template_in_pipeline():
    g = FakeGuided(["Great rep, keep it up.", "Nice controlled pace."])
    fb = CoachPipeline(g, cfg=CFG).rep(rec())
    assert fb.fallback and "knee" in fb.text.lower()
    assert fb.decoding["faithful"] == 0 and fb.decoding["fallback_decode"]
    assert g.fallback_calls == 1
