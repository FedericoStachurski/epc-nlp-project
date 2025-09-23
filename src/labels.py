# src/labels.py
import re

BUCKETS = {
    "insulation": [
        r"\binsulat(?:e|ion|ed)\b", r"\bloft\b", r"\battic\b", r"\broof insulation\b",
        r"\btop[- ]up\b", r"\bcavity (wall )?insulation\b", r"\bsolid wall\b",
        r"\bexternal wall insulation\b", r"\binternal wall insulation\b",
        r"\b(room[- ]in[- ]roof|r-i-r)\b", r"\bfloor insulation\b", r"\bdraught[- ]?proof",
    ],
    "heating": [
        r"\bboiler\b", r"\bcondensing\b", r"\bcombi\b",
        r"\bheating control[s]?\b", r"\bcontrols? upgrade\b",
        r"\bthermostat\b", r"\btrvs?\b|\bthermostatic radiator valves?\b",
        r"\bradiator[s]?\b", r"\bstorage heater[s]?\b", r"\bspace heating\b",
        r"\bheat pump\b|\b(air|ground) source heat pump\b|\b(ASHP|GSHP)\b",
    ],
    "renewables": [
        r"\bsolar\b", r"\bsolar (pv|photovoltaic)\b|\bpv\b|\bphotovoltaic\b",
        r"\bsolar water heating\b|\bsolar thermal\b", r"\bwind turbine\b|\bmicro[- ]?wind\b",
        r"\bmicrogeneration\b",
    ],
    "glazing": [
        r"\bdouble[- ]?glaz(?:ed|ing)\b|\bdg\b|\bdg units?\b",
        r"\btriple[- ]?glaz(?:ed|ing)\b", r"\blow[- ]e\b|\blow emissivit[y|y]\b",
        r"\breplacement (glazing|windows?)\b", r"\bupvc windows?\b",
    ],
    "lighting": [
        r"\blow energy lighting\b", r"\bled\b|\bled (bulbs?|lamps?)\b",
        r"\benergy[- ]saving (bulbs?|lamps?)\b",
    ],
    "hot_water": [
        r"\b(hot )?water (cylinder|tank)\b", r"\bcylinder (thermostat|stat)\b",
        r"\bimmersion (heater)?\b", r"\bcylinder jacket\b|\btank jacket\b",
        r"\bhot water (controls?|upgrades?)\b",
    ],
}

def labels_measure(text: str) -> set[str]:
    s = (text or "").lower()
    # normalise common punctuation/nbspace variants
    s = s.replace("\xa0", " ")
    hits = set()
    for cat, patterns in BUCKETS.items():
        if any(re.search(p, s) for p in patterns):
            hits.add(cat)
    return hits
