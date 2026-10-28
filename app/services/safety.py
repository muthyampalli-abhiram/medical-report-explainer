"""Medical safety and guardrail verification service."""

import re

# Rule definitions with case-insensitive patterns
SAFETY_RULES: list[tuple[str, re.Pattern]] = [
    # Diagnosis statements
    ("diagnosis_you_have", re.compile(r"\byou have\b", re.IGNORECASE)),
    ("diagnosis_you_are_diabetic", re.compile(r"\byou are diabetic\b", re.IGNORECASE)),
    ("diagnosis_you_are_anemic", re.compile(r"\byou are anemic\b", re.IGNORECASE)),
    ("diagnosis_diagnos_root", re.compile(r"diagnos", re.IGNORECASE)),
    ("diagnosis_this_indicates", re.compile(r"\bthis indicates\b", re.IGNORECASE)),
    ("diagnosis_confirms", re.compile(r"\bconfirms\b", re.IGNORECASE)),
    ("diagnosis_suffering_from", re.compile(r"\bsuffering from\b", re.IGNORECASE)),
    # Medication instructions
    ("medication_stop_taking", re.compile(r"\bstop taking\b", re.IGNORECASE)),
    ("medication_start_taking", re.compile(r"\b(start|begin)\s+taking\b", re.IGNORECASE)),
    ("medication_increase_dose", re.compile(r"\bincrease your dose\b", re.IGNORECASE)),
    ("medication_decrease_dose", re.compile(r"\bdecrease your dose\b", re.IGNORECASE)),
    ("medication_take_mg", re.compile(r"\btake\s+\d+(?:\.\d+)?\s*mg\b", re.IGNORECASE)),
    ("medication_you_should_take", re.compile(r"\byou should take\b", re.IGNORECASE)),
    ("medication_no_need", re.compile(r"\bno need for medication\b", re.IGNORECASE)),
    ("medication_skip", re.compile(r"\bskip your medication\b", re.IGNORECASE)),
    # False reassurance or alarm
    ("reassurance_nothing_to_worry", re.compile(r"\bnothing to worry\b", re.IGNORECASE)),
    ("reassurance_dont_worry", re.compile(r"\bdon'?t worry\b", re.IGNORECASE)),
    ("reassurance_do_not_worry", re.compile(r"\bdo\s*not\s+worry\b", re.IGNORECASE)),
    ("reassurance_no_need_doctor", re.compile(r"\bno need to see a doctor\b", re.IGNORECASE)),
    ("reassurance_perfectly_fine", re.compile(r"\bperfectly fine\b", re.IGNORECASE)),
    ("reassurance_nothing_serious", re.compile(r"\bnothing serious\b", re.IGNORECASE)),
    ("reassurance_completely_normal", re.compile(r"\bcompletely normal and safe\b", re.IGNORECASE)),
    ("reassurance_nothing_to_fear", re.compile(r"\bnothing to fear\b", re.IGNORECASE)),
    ("reassurance_flawless", re.compile(r"\b(flawless|zero concern)\b", re.IGNORECASE)),
    ("alarm_life_threatening", re.compile(r"\blife[- ]threatening\b", re.IGNORECASE)),
    ("alarm_you_will_die", re.compile(r"\byou will die\b", re.IGNORECASE)),
    ("alarm_fatal", re.compile(r"\bfatal\b", re.IGNORECASE)),
    ("alarm_emergency_care", re.compile(r"\b(emergency care|extreme emergency|rush to the hospital)\b", re.IGNORECASE)),
    # Extended diagnoses and medication
    ("diagnosis_suggests", re.compile(r"\b(suggests?|point(?:s)?\s+to|indicates?)\s+(?:a\s+|an\s+|early\s+|severe\s+)?(?:diabetes|prediabetes|anemia|disease|hypertension)\b", re.IGNORECASE)),
    ("diagnosis_prediabetic", re.compile(r"\bprediabetic\b", re.IGNORECASE)),
    ("diagnosis_might_have", re.compile(r"\b(might|probably|could)\s+have\b", re.IGNORECASE)),
    ("diagnosis_disease_label", re.compile(r"\b(liver|kidney)\s+disease\b", re.IGNORECASE)),
    ("diagnosis_high_blood_pressure", re.compile(r"\bhigh\s+blood\s+pressure\b", re.IGNORECASE)),
    ("medication_consider_starting", re.compile(r"\bconsider\s+starting\b", re.IGNORECASE)),
    ("medication_prescribe", re.compile(r"\bprescribe\b", re.IGNORECASE)),
    ("medication_reduce_dose", re.compile(r"\b(reduce|decrease|increase)\s+your\s+(dose|insulin|medication|pills?)\b", re.IGNORECASE)),
    ("medication_discontinue", re.compile(r"\bdiscontinue\s+your\b", re.IGNORECASE)),
    ("medication_you_must_take", re.compile(r"\byou\s+must\s+take\b", re.IGNORECASE)),
]


def _normalize_evasion(text: str) -> str:
    """Normalize evasion attempts like hyphens between characters or excess spacing."""
    cleaned = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def check_text(text: str) -> list[str]:
    """Check text against medical safety guardrails. Returns list of violated rule IDs."""
    if not text or not isinstance(text, str):
        return []

    variants = [text]
    norm = _normalize_evasion(text)
    if norm != text:
        variants.append(norm)

    violations: list[str] = []
    for rule_id, pattern in SAFETY_RULES:
        if any(pattern.search(v) for v in variants):
            violations.append(rule_id)

    return violations


def safe_or_fallback(text: str | None, fallback: str) -> tuple[str, bool]:
    """Return text if safe, otherwise return the fallback string with used_fallback=True."""
    if not text or not isinstance(text, str) or not text.strip():
        return fallback, True

    violations = check_text(text)
    if violations:
        return fallback, True

    return text.strip(), False
