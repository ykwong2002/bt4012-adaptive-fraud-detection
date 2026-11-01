"""Prompts and JSON schemas for the two LLM scoring stages. Fields only, never a fraud verdict.
The system prompt is identical for every request so it can be prompt-cached inside a batch."""
from __future__ import annotations

REVIEW_SYSTEM = (
    "You read one online review of a restaurant or hotel and describe its *text evidence* with "
    "numeric fields. You never judge whether the review is fake or genuine. Scores are in [0, 1]. "
    "Definitions: specificity = concrete, checkable details (named dishes, prices, room numbers, "
    "times, staff names, layout); experiential_evidence = a first-hand visit is narrated (sequence "
    "of events, sensory detail, problems encountered); generic_praise = interchangeable superlatives "
    "that could describe any venue; promotional_tone = reads like marketing copy or a call to action; "
    "first_person_detail = personal context (who the reviewer was with, why they came, comparisons "
    "to their other visits); named_concretes = count of distinct named concrete items. "
    "style flags: exclamation_heavy, all_caps_words, template_like (reads like a filled template)."
)

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "specificity": {"type": "number"},
        "experiential_evidence": {"type": "number"},
        "generic_praise": {"type": "number"},
        "promotional_tone": {"type": "number"},
        "first_person_detail": {"type": "number"},
        "named_concretes": {"type": "integer"},
        "exclamation_heavy": {"type": "boolean"},
        "all_caps_words": {"type": "boolean"},
        "template_like": {"type": "boolean"},
    },
    "required": ["specificity", "experiential_evidence", "generic_praise", "promotional_tone",
                 "first_person_detail", "named_concretes", "exclamation_heavy", "all_caps_words",
                 "template_like"],
    "additionalProperties": False,
}
REVIEW_FIELDS = ["specificity", "experiential_evidence", "generic_praise", "promotional_tone",
                 "first_person_detail", "named_concretes", "exclamation_heavy", "all_caps_words",
                 "template_like"]


def review_user_message(text: str, max_chars: int = 3000) -> str:
    t = text if len(text) <= max_chars else text[:max_chars] + " [truncated]"
    return f"<review>\n{t}\n</review>\nReturn the fields."


# ---- pairwise relative fit, one prompt per relation (decisions D4, D7) ----
# relative_fit is the edge weight target. It measures whether the two texts carry the SAME KIND of
# evidence, so that edges joining a generic/promotional review to a detailed first-hand one (the
# camouflage edges) score low, while edges between two reviews of the same kind score high. Two
# templated reviews posted together are usually both fraudulent, so their edge is one we want to KEEP:
# the prompt must never lower the score for similarity, templating or timing (that was v1's bug).
PAIR_PROMPT_VERSION = 2

PAIR_SYSTEM_BASE = (
    "You compare two online reviews of restaurants/hotels that the platform links to each other, and "
    "score how similar they are in the KIND of evidence their text contains. Consider specificity "
    "(concrete, checkable details), first-hand experience (a narrated visit), generic praise "
    "(interchangeable superlatives), promotional tone, and writing style. Output relative_fit in [0, 1]: "
    "1 = the same kind of review (both detailed first-hand accounts, or both generic, promotional or "
    "templated); 0 = opposite kinds (one a detailed first-hand account, the other generic or "
    "promotional). Do not lower relative_fit because the two texts are similar to each other, share "
    "phrasing, follow a template or were posted close together: being the same kind is exactly what "
    "a high score means. Also output consistency in [0, 1] as defined for the relationship, and a "
    "reason of at most 20 words. Never judge whether either review is fake."
)

PAIR_RELATION_DEFS = {
    "rur": "Relationship: SAME AUTHOR. consistency = how plausibly one person wrote both (voice, "
           "vocabulary, level of detail, habits).",
    "rsr": "Relationship: SAME VENUE, SAME STAR RATING. consistency = how compatible the venue facts "
           "in the two texts are (menu, rooms, location, service).",
    "rtr": "Relationship: SAME VENUE, SAME MONTH. consistency = how compatible the venue facts in the "
           "two texts are (menu, rooms, location, service).",
}

PAIR_SCHEMA = {
    "type": "object",
    "properties": {"relative_fit": {"type": "number"}, "consistency": {"type": "number"},
                   "reason": {"type": "string"}},
    "required": ["relative_fit", "consistency", "reason"],
    "additionalProperties": False,
}


def pair_system(relation: str) -> str:
    return PAIR_SYSTEM_BASE + " " + PAIR_RELATION_DEFS[relation]


def pair_user_message(text_a: str, text_b: str, max_chars: int = 2000) -> str:
    a = text_a[:max_chars]; b = text_b[:max_chars]
    return f"<review_a>\n{a}\n</review_a>\n<review_b>\n{b}\n</review_b>\nReturn the fields."
