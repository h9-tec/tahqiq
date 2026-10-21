"""Numeral handling for the numeral probe.

Ported from the Nateq eval protocol (``~/Nateq/eval/normalizer.py``). The probe
asks one question per utterance that contains a number in its reference: does
the hypothesis contain the same digit sequence? Spoken forms and written forms
both map to digits so "خمسة" and "٥" and "5" agree.

The number-word table is deliberately small and Gulf/Egyptian leaning; it is
published and tested, and it only matters for utterances flagged
``has_number``.
"""

from __future__ import annotations

import re

from .tolerant import normalize_tolerant

# After tolerant normalization (ة→ه, hamza folded), so keys are written that way.
NUM_WORDS: dict[str, str] = {
    "صفر": "0",
    "واحد": "1", "وحده": "1", "واحده": "1",
    "اثنين": "2", "ثنين": "2", "ثنتين": "2", "اثنان": "2", "اتنين": "2",
    "ثلاثه": "3", "ثلاث": "3", "تلاته": "3", "تلات": "3",
    "اربعه": "4", "اربع": "4",
    "خمسه": "5", "خمس": "5",
    "سته": "6", "ست": "6",
    "سبعه": "7", "سبع": "7",
    "ثمانيه": "8", "ثمان": "8", "ثمن": "8", "تمانيه": "8", "تمن": "8",
    "تسعه": "9", "تسع": "9",
    "عشره": "10", "عشر": "10",
    "عشرين": "20", "ثلاثين": "30", "تلاتين": "30", "اربعين": "40",
    "خمسين": "50", "ستين": "60", "سبعين": "70", "ثمانين": "80",
    "تمانين": "80", "تسعين": "90",
    "مايه": "100", "مئه": "100", "ميه": "100",
    "الف": "1000",
}

_DIGIT_RE = re.compile(r"\d")
_HAS_DIGIT_RE = re.compile(r"[0-9٠-٩۰-۹]")


def has_written_number(text: str) -> bool:
    """True when the raw text contains any Arabic-Indic, Persian, or Western digit."""
    return bool(_HAS_DIGIT_RE.search(text or ""))


def digit_string(text: str, *, map_number_words: bool = False) -> str:
    """Concatenated digits in reading order, e.g. "رقم ٠٥٠ ١٢٣" → "050123"."""
    t = normalize_tolerant(text)
    if map_number_words:
        t = " ".join(NUM_WORDS.get(w, w) for w in t.split())
    return "".join(_DIGIT_RE.findall(t))
