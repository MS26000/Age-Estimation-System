"""CV approach for the API: pretrained ViT age model + linear probe trained in step 4.

If models/cv_probe.npz exists the probe is used (bracket from logistic regression, exact
age from ridge regression, both stored as plain numpy). Otherwise the pretrained model is
used zero-shot (its 9 age bins mapped to our 6 brackets).
"""
import os
import threading

import numpy as np
from PIL import Image

from . import BRACKETS, BRACKET_EDGES, clamp_age

MODEL_ID = "nateraw/vit-age-classifier"
DEFAULT_PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "cv_probe.npz")


def bin_edges(label):
    if label.lower().startswith("more"):
        return 70.0, 80.0
    a, b = label.split("-")
    return float(a), float(b) + 1.0


def bracket_matrix(bins):
    M = np.zeros((len(bins), len(BRACKETS)))
    for i, (blo, bhi) in enumerate(bins):
        for j, (lo, hi) in enumerate(BRACKET_EDGES):
            M[i, j] = max(0.0, min(bhi, hi) - max(blo, lo)) / (bhi - blo)
    return M


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def probe_predict(emb, probe):
    """emb: (d,) ViT embedding -> dict(bracket, age, confidence). Pure numpy."""
    x = (emb - probe["mean"]) / probe["scale"]
    p = _softmax(probe["clf_w"] @ x + probe["clf_b"])
    k = int(p.argmax())
    bracket = str(probe["classes"][k])
    age = clamp_age(float(probe["reg_w"] @ x + probe["reg_b"]), bracket)
    return {"bracket": bracket, "age": age, "confidence": float(p[k])}


def zero_shot_predict(probs, M, mids):
    bp = probs @ M
    k = int(bp.argmax())
    bracket = BRACKETS[k]
    return {"bracket": bracket, "age": clamp_age(float(probs @ mids), bracket), "confidence": float(bp[k])}


class CVAgeModel:
    def __init__(self, probe_path=DEFAULT_PROBE, model_id=MODEL_ID, device="cpu"):
        import torch
        from transformers import AutoImageProcessor, ViTForImageClassification
        self._torch = torch
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = ViTForImageClassification.from_pretrained(model_id).to(device).eval()
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        bins = [bin_edges(id2label[i]) for i in range(len(id2label))]
        self.mids = np.array([(a + b) / 2 for a, b in bins])
        self.M = bracket_matrix(bins)
        self.probe = None
        probe_path = os.path.abspath(probe_path)
        if os.path.exists(probe_path):
            z = np.load(probe_path, allow_pickle=True)
            self.probe = {k: z[k] for k in ["mean", "scale", "clf_w", "clf_b", "classes", "reg_w", "reg_b"]}
        self._lock = threading.Lock()

    @property
    def mode(self):
        return "probe" if self.probe is not None else "zeroshot"

    def predict(self, face_bgr):
        torch = self._torch
        img = Image.fromarray(np.ascontiguousarray(face_bgr[:, :, ::-1]))
        with self._lock, torch.no_grad():
            inp = self.proc(images=img, return_tensors="pt")
            h = self.model.vit(pixel_values=inp["pixel_values"].to(self.device)).last_hidden_state[:, 0]
            probs = self.model.classifier(h).softmax(-1)[0].float().cpu().numpy()
            emb = h[0].float().cpu().numpy()
        if self.probe is not None:
            return probe_predict(emb, self.probe)
        return zero_shot_predict(probs, self.M, self.mids)
