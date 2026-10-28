# Integrating PhysioCoach-NLP into PhysioLens (after the course)

PhysioCoach-NLP is a separate service. PhysioLens does not import its code; it
sends one HTTP request per finished rep, and one per finished session. This
keeps the environments separate (PhysioLens's MediaPipe/TensorFlow protobuf pins
never meet PyTorch).

## 1. Run the service

```bash
cd physiocoach-nlp
physiocoach serve --port 8000
```

## 2. Map PhysioLens output to a rep record

Your CNN-LSTM has independent sigmoid heads for `SHALLOW` and `KNEE_BENT`, and
your 22-feature vector contains the joint angles. Send only the faults you
actually measure. PhysioCoach will never praise or mention a fault that is
missing from `faults`.

```python
import requests

COACH_URL = "http://127.0.0.1:8000/api/feedback/rep"


def coach_rep(probs, feats, side, set_no, rep_no, lang="en", speak=True):
    record = {
        "exercise": "side_leg_raise",
        "side": side,                       # "left" or "right"
        "set": set_no,
        "rep": rep_no,
        "faults": {
            "SHALLOW": float(probs["SHALLOW"]),
            "KNEE_BENT": float(probs["KNEE_BENT"]),
        },
        "features": {
            "target_hip_abduction_deg": 40.0,                       # your target
            "peak_hip_abduction_deg": float(feats["peak_hip_abd"]),   # from your feature vector
            "knee_angle_deg": float(feats["min_knee_angle"]),
        },
    }
    r = requests.post(COACH_URL, json={"record": record, "lang": lang, "backend": "t5", "speak": speak},
                      timeout=3)
    r.raise_for_status()
    return r.json()   # {"text": ..., "audio_url": "/audio/....wav", ...}
```

Call it in `main_with_model` where a rep window closes, **in a background
thread**, so the video loop never waits on the network:

```python
import threading

def on_rep_finished(probs, feats, side, set_no, rep_no):
    def work():
        fb = coach_rep(probs, feats, side, set_no, rep_no)
        show_text_overlay(fb["text"])
        if fb["audio_url"]:
            play_wav("http://127.0.0.1:8000" + fb["audio_url"])
    threading.Thread(target=work, daemon=True).start()
```

At the end of a session, send all rep records to `/api/feedback/session` as
`{"session": {"exercise": ..., "side": ..., "reps": [...]}}`.

## 3. Names must match the config

If your feature or fault names differ, change
`src/physiocoach/config/exercises.yaml` (not your FYP code), then run `pytest`.
The deficit in "about 12 degrees short" is computed from the two features named
under the fault's `deficit:` key.

## 4. Latency budget

| Step | Typical (laptop CPU) |
|---|---|
| Flan-T5-small cue (PyTorch) | about 100–300 ms; less with ONNX |
| Faithfulness check | under 1 ms |
| NLLB translation | about 0.5–1.5 s first time, instant when cached |
| MMS speech | about 0.3–1 s first time, instant when cached |

Measure your own numbers with `scripts/evaluate.py` and
`scripts/speech_eval.py`. Cues repeat a lot, so after a few sessions most
requests hit the caches.
