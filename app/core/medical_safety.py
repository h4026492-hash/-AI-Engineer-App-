"""Conservative, deterministic safety gates for the public health-education demo.

These checks are intentionally narrow and are not a clinical triage system. They
route obvious emergency and individualized-medication questions to a human
professional before retrieval or model generation.
"""

from __future__ import annotations

import re

EDUCATIONAL_FOOTER = (
    "\n\nEducational information only—not a diagnosis or treatment plan. "
    "Confirm personal health questions with a qualified health professional."
)

EMERGENCY_REPLY = (
    "If this is happening now to you or someone nearby, call 911 (or your local "
    "emergency number) now. This demo cannot assess emergencies; do not wait for "
    "an AI response."
)

MEDICATION_REPLY = (
    "I can explain general medicine-label terms, but I can't check whether a "
    "medicine is safe for a specific person, compare personal drug interactions, "
    "recommend a dose, or tell anyone to start, stop, or change a medicine. "
    "Please contact the pharmacist or clinician who knows the full medication "
    "list. If symptoms are severe or this may be an emergency, call 911."
)

PERSONAL_LAB_REPLY = (
    "I can't diagnose or interpret a specific person's lab result. Reference "
    "ranges and results need to be considered with the person's health history, "
    "symptoms, and other findings. Please contact the clinician who ordered the "
    "test for personal guidance. If this may be an emergency, call 911."
)

_PERSONAL_NOW = re.compile(
    r"\b(?:i(?:'m| am| have|'ve)?|my|me|someone|somebody|my child|my parent|"
    r"my mom|my dad|my husband|my wife|my partner|he is|she is|they are|"
    r"dad|mom|father|mother|grandpa|grandma|grandfather|grandmother|brother|sister|"
    r"child|baby|spouse|partner|right now|currently|just happened|what should i do|help)\b",
    re.IGNORECASE,
)
_URGENT_SYMPTOM = re.compile(
    r"\b(?:chest pain|pain in (?:the|my) chest|trouble breathing|difficulty breathing|"
    r"shortness of breath|can't breathe|cannot breathe|stroke|face droop(?:ing)?|"
    r"sudden weakness|severe allergic reaction|anaphylaxis|overdose|unconscious|"
    r"seizure|suicid(?:e|al)|self[- ]harm|bleeding won't stop|severe bleeding)\b",
    re.IGNORECASE,
)
_MEDICATION = re.compile(
    r"\b(?:medicine|medicines|meds|medication|medications|drug|drugs|pill|pills|"
    r"tablets?|prescriptions?|dose|dosage|ibuprofen|warfarin|aspirin|acetaminophen|tylenol|"
    r"advil|blood thinner|supplements?)\b",
    re.IGNORECASE,
)
_DOSE_REQUEST = re.compile(
    r"\b(?:dose|dosage|how much|how many|mg|milligram|milliliter|ml|missed dose|"
    r"forgot(?:ten)? (?:a )?dose|double (?:a )?dose)\b",
    re.IGNORECASE,
)
_SPECIFIC_INTERACTION = re.compile(
    r"\b(?:interact(?:ion)?s?|combine|mix|take|takes|taking|taken|use|uses|using|used)"
    r"\b.*\b(?:with|and|between)\b|"
    r"\b(?:with|between)\b.*\b(?:interact(?:ion)?s?|combine|mix)\b",
    re.IGNORECASE,
)
_PERSONAL = re.compile(
    r"\b(?:i|i'm|i am|my|me|we|our|my child|my parent|my mom|my dad|"
    r"my husband|my wife|my partner|my grandfather|my grandmother)\b",
    re.IGNORECASE,
)
_LAB = re.compile(
    r"\b(?:lab|laboratory|bloodwork|blood work|test result|results?|cbc|cmp|"
    r"a1c|hemoglobin|cholesterol|creatinine|glucose|inr)\b",
    re.IGNORECASE,
)
_DIAGNOSIS_ASK = re.compile(
    r"\b(?:do i have|could i have)\s+(?:a |an |the )?(?:disease|condition|diabetes|"
    r"cancer|infection|heart attack|stroke|illness|anemia)\b|"
    r"\b(?:am i sick|what disease do i have|diagnose me|is this cancer|"
    r"is this a heart attack)\b",
    re.IGNORECASE,
)


def screen_medical_question(question: str) -> str | None:
    """Return a safe handoff response when a question needs a human professional.

    The check is a fail-safe convenience, not a comprehensive safety detector.
    The UI separately displays emergency guidance on every page.
    """
    if _URGENT_SYMPTOM.search(question) and _PERSONAL_NOW.search(question):
        return EMERGENCY_REPLY

    if _DIAGNOSIS_ASK.search(question):
        return PERSONAL_LAB_REPLY

    is_personal_lab = _PERSONAL.search(question) and _LAB.search(question)
    if is_personal_lab:
        return PERSONAL_LAB_REPLY

    mentions_medicine = _MEDICATION.search(question) is not None
    needs_dosing_handoff = _DOSE_REQUEST.search(question) is not None
    asks_about_specific_interaction = _SPECIFIC_INTERACTION.search(question) is not None
    needs_medication_handoff = mentions_medicine and (
        _PERSONAL.search(question) is not None
        or needs_dosing_handoff
        or asks_about_specific_interaction
    )
    if needs_medication_handoff:
        return MEDICATION_REPLY

    return None
