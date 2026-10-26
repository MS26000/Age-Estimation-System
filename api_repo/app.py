"""Live API: POST /predict {"url": "<Wikipedia page link>"} -> age bracket at the time of the photo.

Flow: parse link -> Wikipedia API (live) -> lead image (+ name-matched page images as fallback)
-> Commons photo date -> download 960px thumbnail -> face rule -> model -> JSON.
If no photo passes the face rule the API says so (status "no_usable_photo") instead of guessing.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agepred import wiki
from agepred.faces import REASON_TEXT, FaceDetector, decode_image, encode_jpeg

log = logging.getLogger("agepred")
logging.basicConfig(level=logging.INFO)

STATE = {"detector": None, "cv": None, "llm": None, "default": "cv"}


def load_models():
    from agepred.cv_model import CVAgeModel
    from agepred.llm import GeminiAgeEstimator
    STATE["detector"] = FaceDetector()
    STATE["cv"] = CVAgeModel()
    try:
        STATE["llm"] = GeminiAgeEstimator.from_env()
    except Exception as e:  # missing SDK etc.: keep the CV approach working
        log.warning("LLM approach disabled: %s", e)
        STATE["llm"] = None
    default = os.environ.get("DEFAULT_APPROACH", "cv").lower()
    STATE["default"] = default if default == "cv" or (default == "llm" and STATE["llm"]) else "cv"
    log.info("Loaded: cv mode=%s, llm=%s, default=%s", STATE["cv"].mode, bool(STATE["llm"]), STATE["default"])


@asynccontextmanager
async def lifespan(app):
    load_models()
    yield


app = FastAPI(title="Age at the time of the photo", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    url: str
    approach: Optional[str] = None  # "cv" or "llm"; default is set by the server


def _no_photo(person, reasons):
    texts = list(dict.fromkeys(REASON_TEXT.get(r, r) for r in reasons)) or ["the page has no usable image"]
    return {"person": person, "status": "no_usable_photo",
            "message": "No usable photo found: " + "; ".join(texts) + ".",
            "reasons": list(dict.fromkeys(reasons)),
            "image_used": None, "photo_date": None,
            "predicted_age_bracket": None, "predicted_age": None, "confidence": None}


@app.get("/")
def root():
    return {"service": "Age at the time of the photo",
            "usage": 'POST /predict with {"url": "https://en.wikipedia.org/wiki/<Person_Name>"}',
            "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    ok = STATE["cv"] is not None and STATE["detector"] is not None
    return {"status": "ok" if ok else "loading",
            "approaches": ["cv"] + (["llm"] if STATE["llm"] else []),
            "default_approach": STATE["default"],
            "cv_mode": STATE["cv"].mode if STATE["cv"] else None}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    if STATE["cv"] is None or STATE["detector"] is None:
        raise HTTPException(503, "Models are still loading, please retry in a minute")
    approach = (req.approach or STATE["default"]).lower()
    if approach not in ("cv", "llm"):
        raise HTTPException(400, 'approach must be "cv" or "llm"')
    if approach == "llm" and STATE["llm"] is None:
        raise HTTPException(400, 'The "llm" approach is not enabled on this server; use "cv"')

    try:
        host, title = wiki.parse_wikipedia_url(req.url)
        page = wiki.get_page_info(host, title)
    except wiki.BadURL as e:
        raise HTTPException(400, str(e))
    except wiki.PageNotFound as e:
        raise HTTPException(404, str(e))
    except wiki.UpstreamError as e:
        raise HTTPException(502, str(e))

    try:
        cands = wiki.resolve_candidates(host, page)
    except wiki.UpstreamError as e:
        raise HTTPException(502, str(e))
    if not cands:
        return _no_photo(page["title"], ["no_lead_image"])

    reasons, chosen, face = [], None, None
    for c in cands:
        try:
            img = decode_image(wiki.download_image(c["thumb_url"]))
        except wiki.UpstreamError:
            reasons.append("download_failed")
            continue
        if img is None:
            reasons.append("unreadable_image")
            continue
        res = STATE["detector"].detect_main_face(img)
        if res["ok"]:
            chosen, face = c, res
            break
        reasons.append(res["reason"])
    if chosen is None:
        return _no_photo(page["title"], reasons)

    warnings, used = [], approach
    if approach == "llm":
        try:
            pred = STATE["llm"].estimate(encode_jpeg(face["crop"]))
        except Exception as e:  # noqa: BLE001  quota, network, bad answer ...
            log.warning("LLM failed, falling back to CV: %s", e)
            warnings.append("llm_failed_fell_back_to_cv")
            used, pred = "cv", STATE["cv"].predict(face["crop"])
    else:
        pred = STATE["cv"].predict(face["crop"])

    out = {"person": page["title"],
           "status": "ok",
           "image_used": chosen["image_url"],
           "image_page": chosen["file_page"],
           "image_source": chosen["source"],
           "photo_date": chosen["photo_date"],
           "predicted_age_bracket": pred["bracket"],
           "predicted_age": pred["age"],
           "confidence": round(pred["confidence"], 2),
           "approach": used}
    if warnings:
        out["warnings"] = warnings
    log.info("%s -> %s (%s) in %.1fs", page["title"], pred["bracket"], used, time.time() - t0)
    return out
