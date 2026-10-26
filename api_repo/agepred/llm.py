"""Optional LLM approach for the API (Gemini). Enabled only if GEMINI_API_KEY is set as a
secret on the server. The key is read from the environment and never stored in the repo.
Same prompt and parsing as pipeline/step5_llm_model.py."""
import json
import os
import re

from . import BRACKETS, bracket_of_age, clamp_age

PROMPT = (
    "You are an expert at estimating a person's age from a photograph of their face. "
    "Estimate how old the person was when this photo was taken. "
    "Use only what is visible in the image. "
    'Reply with ONLY a JSON object and no other text: '
    '{"age": <integer number of years>, '
    '"bracket": "<exactly one of: Under 18, 18-24, 25-34, 35-44, 45-54, 55+>", '
    '"confidence": <number between 0 and 1>}. '
    "The bracket must contain the age."
)


def parse_answer(raw):
    m = re.search(r"\{.*?\}", raw or "", re.S)
    if not m:
        raise ValueError("LLM answer has no JSON")
    obj = json.loads(m.group(0))
    age = float(obj["age"])
    if not 0 <= age <= 110:
        raise ValueError("age out of range")
    b = str(obj.get("bracket", "")).replace("\u2013", "-").replace("\u2014", "-").strip()
    bracket = b if b in BRACKETS else bracket_of_age(age)
    conf = min(max(float(obj.get("confidence", 0.5)), 0.0), 1.0)
    return {"bracket": bracket, "age": clamp_age(age, bracket), "confidence": conf}


class GeminiAgeEstimator:
    def __init__(self, api_key, model):
        from google import genai
        from google.genai import types
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model = model

    @classmethod
    def from_env(cls):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            return None
        return cls(key, os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite"))

    def estimate(self, jpeg_bytes):
        t = self._types
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[t.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"), PROMPT],
            config=t.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"))
        return parse_answer(resp.text)
