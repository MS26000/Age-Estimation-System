"""Offline tests (no internet, no torch needed):  pytest -q tests/test_offline.py

They check the parts that must match between evaluation and serving:
the numpy probe equals the scikit-learn model, the face rule, the date parser, URL parsing,
and the API logic with the network and models replaced by fakes.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agepred import BRACKETS, bracket_of_age, clamp_age, wiki  # noqa: E402
from agepred.cv_model import bin_edges, bracket_matrix, probe_predict, zero_shot_predict  # noqa: E402


# ------------------------------------------------------------------ helpers
def test_bracket_and_clamp():
    assert [bracket_of_age(a) for a in (0, 17, 18, 24, 25, 34, 35, 44, 45, 54, 55, 99)] == \
        ["Under 18", "Under 18", "18-24", "18-24", "25-34", "25-34", "35-44", "35-44", "45-54", "45-54", "55+", "55+"]
    assert clamp_age(24.6, "18-24") == 24 and clamp_age(40, "25-34") == 34 and clamp_age(3, "55+") == 55


def test_zero_shot_mapping_rows_sum_to_one():
    labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
    bins = [bin_edges(l) for l in labels]
    M = bracket_matrix(bins)
    assert np.allclose(M.sum(1), 1)
    probs = np.zeros(9); probs[3] = 1.0                      # model says "20-29"
    out = zero_shot_predict(probs, M, np.array([(a + b) / 2 for a, b in bins]))
    assert out["bracket"] in ("18-24", "25-34") and abs(out["confidence"] - 0.5) < 1e-9


def test_numpy_probe_matches_sklearn(tmp_path):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(0)
    X = rng.randn(600, 32); ages = rng.uniform(5, 80, 600)
    X[:, 0] += ages / 20                                      # signal
    y = np.array([bracket_of_age(a) for a in ages])
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(Xs, y)
    reg = Ridge(alpha=10).fit(Xs, ages)
    path = tmp_path / "probe.npz"                              # same keys/format as step 4 writes
    np.savez(path, mean=sc.mean_, scale=sc.scale_, clf_w=clf.coef_, clf_b=clf.intercept_,
             classes=np.array(clf.classes_), reg_w=reg.coef_, reg_b=np.array(reg.intercept_))
    z = np.load(path, allow_pickle=True)
    probe = {k: z[k] for k in ["mean", "scale", "clf_w", "clf_b", "classes", "reg_w", "reg_b"]}
    P = clf.predict_proba(Xs)
    for i in range(50):
        out = probe_predict(X[i], probe)
        assert out["bracket"] == clf.classes_[P[i].argmax()]
        assert abs(out["confidence"] - P[i].max()) < 1e-6
        assert out["age"] == clamp_age(reg.predict(Xs[i:i + 1])[0], out["bracket"])


# ------------------------------------------------------------------ dates / URLs
@pytest.mark.parametrize("raw,expected", [
    ("2016-05-03", "2016-05-03"), ("2013-06-21 19:02:58", "2013-06-21"),
    ("Taken on 6 April 2006", "2006-04-06"), ("30 March 2015, 18:16:28", "2015-03-30"),
    ("20 September 2025 (according to Exif data)", "2025-09-20"),
    ("March 2012", "2012-03"), ("1985", "1985"),
    ("circa 1990", None), ("23 April 2008 (original upload date)", None), ("10/6/2006", None),
    ("1972-1973 season", None), (None, None)])
def test_photo_date_rules(raw, expected):
    lo, hi, prec = wiki.parse_photo_date(raw)
    if lo is None and prec == "unparseable_date":
        c = wiki.clean_raw(raw)
        lo, hi, prec = wiki.parse_photo_date(c) if c else (None, None, "rejected")
    got = wiki.format_photo_date(lo, hi, prec) if lo is not None else None
    assert got == expected


def test_url_parsing():
    assert wiki.parse_wikipedia_url("https://en.wikipedia.org/wiki/Tom_Hanks") == ("en.wikipedia.org", "Tom Hanks")
    assert wiki.parse_wikipedia_url("https://de.m.wikipedia.org/wiki/Angela_Merkel#Leben")[0] == "de.wikipedia.org"
    for bad in ["https://example.com/wiki/X", "nonsense", "https://en.wikipedia.org/", "https://evil.com/?x=en.wikipedia.org/wiki/X"]:
        with pytest.raises(wiki.BadURL):
            wiki.parse_wikipedia_url(bad)


# ------------------------------------------------------------------ face rule with a stubbed detector
def _fake_detector(rows):
    import cv2
    from agepred import faces

    class Stub:
        def setInputSize(self, s): pass
        def detect(self, img): return 1, (np.array(rows, dtype=np.float32) if rows else None)

    d = faces.FaceDetector.__new__(faces.FaceDetector)
    import threading
    d._det, d._lock = Stub(), threading.Lock()
    return d


def _face(x, y, w, h, score=0.95):
    return [x, y, w, h] + [0] * 10 + [score]


def test_face_rule():
    img = np.full((800, 800, 3), 128, np.uint8)
    assert _fake_detector([]).detect_main_face(img)["reason"] == "no_face"
    assert _fake_detector([_face(100, 100, 200, 200, 0.5)]).detect_main_face(img)["reason"] == "no_face"
    assert _fake_detector([_face(100, 100, 200, 200), _face(500, 100, 180, 180)]).detect_main_face(img)["reason"] == "multiple_faces"
    assert _fake_detector([_face(100, 100, 40, 40)]).detect_main_face(img)["reason"] == "face_too_small"
    ok = _fake_detector([_face(100, 100, 200, 200), _face(600, 600, 60, 60)]).detect_main_face(img)   # 200 >= 2*60
    assert ok["ok"] and ok["crop"].shape[0] == 320 and ok["crop"].shape[1] == 320                       # 200 + 2*30%


# ------------------------------------------------------------------ API with fakes
class FakeDetector:
    def __init__(self, result): self.result = result
    def detect_main_face(self, img): return self.result


class FakeCV:
    mode = "probe"
    def predict(self, crop): return {"bracket": "35-44", "age": 38, "confidence": 0.71}


class FakeLLM:
    def __init__(self, fail=False): self.fail = fail
    def estimate(self, jpeg):
        if self.fail: raise RuntimeError("quota")
        return {"bracket": "25-34", "age": 29, "confidence": 0.6}


@pytest.fixture()
def client(monkeypatch):
    import cv2
    from fastapi.testclient import TestClient
    import app as appmod
    ok_crop = np.full((200, 200, 3), 100, np.uint8)
    appmod.STATE.update(detector=FakeDetector({"ok": True, "reason": None, "crop": ok_crop}),
                        cv=FakeCV(), llm=None, default="cv")
    page = {"title": "Jane Doe", "lead_file": "Jane_Doe.jpg", "lead_thumb": None, "lead_original": None, "images": []}
    cand = {"file": "Jane_Doe.jpg", "source": "lead_image", "thumb_url": "https://upload.wikimedia.org/x.jpg",
            "image_url": "https://upload.wikimedia.org/orig.jpg", "file_page": "https://commons.wikimedia.org/wiki/File:Jane_Doe.jpg",
            "photo_date": "2016-05-03"}
    jpg = cv2.imencode(".jpg", np.full((300, 300, 3), 90, np.uint8))[1].tobytes()
    monkeypatch.setattr(appmod.wiki, "get_page_info", lambda host, title: page)
    monkeypatch.setattr(appmod.wiki, "resolve_candidates", lambda host, p: [cand])
    monkeypatch.setattr(appmod.wiki, "download_image", lambda url: jpg)
    return TestClient(appmod.app), appmod


def test_predict_ok(client):
    c, _ = client
    r = c.post("/predict", json={"url": "https://en.wikipedia.org/wiki/Jane_Doe"})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["person"] == "Jane Doe" and j["photo_date"] == "2016-05-03"
    assert j["predicted_age_bracket"] == "35-44" and j["predicted_age"] == 38 and j["confidence"] == 0.71
    assert j["image_used"].endswith("orig.jpg") and j["approach"] == "cv"


def test_predict_no_usable_photo(client):
    c, app = client
    app.STATE["detector"] = FakeDetector({"ok": False, "reason": "multiple_faces", "crop": None})
    j = c.post("/predict", json={"url": "https://en.wikipedia.org/wiki/Jane_Doe"}).json()
    assert j["status"] == "no_usable_photo" and j["predicted_age_bracket"] is None
    assert "several people" in j["message"]


def test_predict_no_image_at_all(client, monkeypatch):
    c, app = client
    monkeypatch.setattr(app.wiki, "resolve_candidates", lambda host, p: [])
    j = c.post("/predict", json={"url": "https://en.wikipedia.org/wiki/Jane_Doe"}).json()
    assert j["status"] == "no_usable_photo" and j["reasons"] == ["no_lead_image"]


def test_errors(client, monkeypatch):
    c, app = client
    assert c.post("/predict", json={"url": "https://example.com/wiki/X"}).status_code == 400
    assert c.post("/predict", json={"url": "https://en.wikipedia.org/wiki/Jane_Doe", "approach": "llm"}).status_code == 400
    assert c.post("/predict", json={"url": "https://en.wikipedia.org/wiki/Jane_Doe", "approach": "xyz"}).status_code == 400
    def missing(host, title): raise app.wiki.PageNotFound("Wikipedia page not found: X")
    monkeypatch.setattr(app.wiki, "get_page_info", missing)
    assert c.post("/predict", json={"url": "https://en.wikipedia.org/wiki/Nobody_Here"}).status_code == 404


def test_llm_and_fallback(client):
    c, app = client
    app.STATE["llm"] = FakeLLM()
    j = c.post("/predict", json={"url": "https://en.wikipedia.org/wiki/Jane_Doe", "approach": "llm"}).json()
    assert j["approach"] == "llm" and j["predicted_age_bracket"] == "25-34"
    app.STATE["llm"] = FakeLLM(fail=True)
    j = c.post("/predict", json={"url": "https://en.wikipedia.org/wiki/Jane_Doe", "approach": "llm"}).json()
    assert j["approach"] == "cv" and j["warnings"] == ["llm_failed_fell_back_to_cv"]


def test_health_and_root(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"
    assert "usage" in c.get("/").json()


# ------------------------------------------------------------------ Commons returns odd extmetadata shapes
@pytest.mark.parametrize("extmeta,expected", [
    ({"DateTimeOriginal": {"value": "2016-05-03"}}, "2016-05-03"),
    ({}, None), ([], None), (None, None), ({"DateTimeOriginal": []}, None)])
def test_commons_extmetadata_shapes(monkeypatch, extmeta, expected):
    page = {"title": "File:A.jpg", "imageinfo": [{"url": "u", "thumburl": "t", "descriptionurl": "d",
                                                  "mime": "image/jpeg", "extmetadata": extmeta}]}
    monkeypatch.setattr(wiki, "_get_json", lambda url, params, **kw: {"query": {"pages": [page]}})
    assert wiki.commons_info(["A.jpg"])["A.jpg"]["raw_date"] == expected
