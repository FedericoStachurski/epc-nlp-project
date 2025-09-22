import re

# simple keyword buckets you can refine later
BUCKETS = {
    "insulation": [
        r"\binsulation\b", r"\bloft\b", r"\bcavity\b", r"\bsolid wall\b",
        r"\broom[- ]in[- ]roof\b", r"\bfloor insulation\b", r"\bjack(et)?\b",
    ],
    "heating": [
        r"\bboiler\b", r"\bcondensing\b", r"\bheating controls?\b",
        r"\bradiator(s)?\b", r"\bthermostat\b", r"\bstorage heater(s)?\b",
        r"\bheat pump\b",
    ],
    "renewables": [
        r"\bsolar\b", r"\bphotovoltaic\b", r"\bsolar water\b", r"\bwind turbine\b",
    ],
    "glazing": [
        r"\bdouble glazing\b", r"\blow[- ]e\b", r"\breplacement glazing\b",
        r"\btriple glazing\b", r"\bglazed\b",
    ],
    "lighting": [
        r"\blow energy lighting\b", r"\bled\b"
    ],
    "hot_water": [
        r"\b(hot )?water cylinder\b", r"\bimmersion\b"
    ],
}

def label_measure(text: str) -> str | None:
    s = (text or "").lower()
    for cat, patterns in BUCKETS.items():
        for pat in patterns:
            if re.search(pat, s):
                return cat
    return None
