"""Shared constants/helpers for the live API.

Keep in sync with pipeline/step3 (face rule) and pipeline/step4 (bracket edges, clamp).
"""
BRACKETS = ["Under 18", "18-24", "25-34", "35-44", "45-54", "55+"]
BRACKET_EDGES = [(0, 18), (18, 25), (25, 35), (35, 45), (45, 55), (55, 100)]  # [lo, hi)


def bracket_of_age(age):
    for name, (lo, hi) in zip(BRACKETS, BRACKET_EDGES):
        if lo <= age < hi:
            return name
    return BRACKETS[-1]


def clamp_age(age, bracket):
    """Make the exact age consistent with the predicted bracket."""
    lo, hi = BRACKET_EDGES[BRACKETS.index(bracket)]
    return int(round(min(max(float(age), lo), hi - 1)))
