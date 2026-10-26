"""Face rule used by the API. It mirrors pipeline/step3_faces_and_split.py exactly:
YuNet detector, score >= 0.8, one clear main face (>= 2x as wide as any other face,
>= 60 px), crop with a 30% margin. Photos that fail the rule are reported as unusable.
"""
import os
import threading

import cv2
import numpy as np
import requests

SCORE_MIN = 0.8
MIN_FACE_PX = 60
DOMINANCE = 2.0
MARGIN = 0.3
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "yunet.onnx")

REASON_TEXT = {
    "no_lead_image": "the page has no usable image",
    "download_failed": "the image could not be downloaded",
    "unreadable_image": "the image could not be decoded",
    "no_face": "no face was detected in the photo",
    "multiple_faces": "the photo shows several people, so it is unclear which one is the page's subject",
    "face_too_small": "the face in the photo is too small to estimate an age",
}


def decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class FaceDetector:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            r = requests.get(MODEL_URL, timeout=60)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(r.content)
        if os.path.getsize(model_path) < 100_000:
            raise RuntimeError(f"{model_path} looks wrong (too small). Re-download the YuNet ONNX model.")
        self._det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), SCORE_MIN, 0.3, 5000)
        self._lock = threading.Lock()  # the OpenCV detector keeps state (input size)

    def detect_main_face(self, img):
        """Return dict(ok, reason, crop, n_faces, box, score). crop is a BGR array."""
        h, w = img.shape[:2]
        with self._lock:
            self._det.setInputSize((w, h))
            _, faces = self._det.detect(img)
        faces = [] if faces is None else list(faces)
        real = sorted([f for f in faces if f[14] >= SCORE_MIN], key=lambda f: f[2] * f[3], reverse=True)
        base = {"ok": False, "reason": None, "crop": None, "n_faces": len(faces), "box": None, "score": None}
        if not real:
            return {**base, "reason": "no_face"}
        main = real[0]
        if len(real) > 1 and main[2] < DOMINANCE * real[1][2]:
            return {**base, "reason": "multiple_faces"}
        if main[2] < MIN_FACE_PX or main[3] < MIN_FACE_PX:
            return {**base, "reason": "face_too_small"}
        x, y, fw, fh = [float(v) for v in main[:4]]
        mx, my = fw * MARGIN, fh * MARGIN
        x0, y0 = int(max(0, x - mx)), int(max(0, y - my))
        x1, y1 = int(min(w, x + fw + mx)), int(min(h, y + fh + my))
        crop = img[y0:y1, x0:x1]
        # JPEG round trip so the model sees the same pixels as in training (step 3 saved JPEG crops)
        crop = decode_image(encode_jpeg(crop))
        return {**base, "ok": True, "crop": crop, "box": (x, y, fw, fh), "score": float(main[14])}
