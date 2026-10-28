from fastapi.testclient import TestClient

from physiocoach.api import create_app
from physiocoach.factory import Registry, Settings

client = TestClient(create_app(Registry(settings=Settings(translator="none", tts="none", t5_dir="does/not/exist"))))


def test_health_and_exercises():
    assert client.get("/api/health").json()["status"] == "ok"
    ex = client.get("/api/exercises").json()
    assert {"side_leg_raise", "squat", "shoulder_abduction"} <= set(ex)


def test_rep_feedback_english_and_urdu():
    rec = client.get("/api/sample/rep", params={"exercise": "side_leg_raise", "p_fault": 0.9, "seed": 1}).json()
    en = client.post("/api/feedback/rep", json={"record": rec}).json()
    assert en["faithful_raw"] and en["text"]
    ur = client.post("/api/feedback/rep", json={"record": rec, "lang": "ur"}).json()
    assert ur["lang"] == "ur" and ur["text"] != ur["text_en"]


def test_session_feedback():
    s = client.get("/api/sample/session", params={"exercise": "squat", "seed": 2}).json()
    r = client.post("/api/feedback/session", json={"session": s})
    assert r.status_code == 200 and r.json()["kind"] == "session"


def test_errors_are_clear():
    bad = client.post("/api/feedback/rep", json={"record": {"exercise": "yoga", "faults": {}}})
    assert bad.status_code == 422
    rec = client.get("/api/sample/rep", params={"seed": 3}).json()
    missing = client.post("/api/feedback/rep", json={"record": rec, "backend": "t5"})
    assert missing.status_code == 503 and "train" in missing.json()["detail"].lower()
    no_tts = client.post("/api/feedback/rep", json={"record": rec, "speak": True})
    assert no_tts.status_code == 503


def test_web_page_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "PhysioCoach-NLP" in r.text


def test_compare_endpoint_annotates_and_reports_unavailable():
    rec = client.get("/api/sample/rep", params={"exercise": "squat", "p_fault": 0.9, "seed": 4}).json()
    out = client.post("/api/compare", json={"record": rec, "backends": ["template", "t5"]}).json()
    assert out["facts"]["kind"] == "rep" and out["facts"]["present"]
    tmpl, t5 = out["results"]
    assert tmpl["available"] and tmpl["faithful"] and tmpl["annotation"]["sentences"]
    assert not t5["available"] and "train" in t5["error"].lower()


def test_compare_session_and_validation():
    s = client.get("/api/sample/session", params={"exercise": "shoulder_abduction", "seed": 5}).json()
    out = client.post("/api/compare", json={"session": s, "backends": ["template"]}).json()
    assert out["facts"]["kind"] == "session" and out["results"][0]["faithful"]
    assert client.post("/api/compare", json={"backends": ["template"]}).status_code == 422


def test_new_pages_are_served():
    for path, marker in (("/compare", "Compare generators"), ("/live", "Live session"), ("/static/style.css", "--accent")):
        r = client.get(path)
        assert r.status_code == 200 and marker in r.text, path
