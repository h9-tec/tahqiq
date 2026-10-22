"""Variant-tolerant Arabic normalizer.

Ported from the Saudi TTS bench (``bench/normalize.py``) and extended with an
optional spelling-variant table (``data/variants.yaml``). The goal is to stop
counting orthographic choices as recognition errors: dialectal Arabic has no
standard spelling, and the same transcriber writes the same word two ways.

Pipeline (fixed order; every step has a reason):
  1. NFC                         same glyph as one code point
  2. strip tashkeel              vowel marks are not transcribed consistently
  3. strip tatweel (ـ)           decorative
  4. alef unify   ا أ إ آ ٱ → ا
  5. alef-maksura → ya  ى → ي
  6. ta-marbuta → ha    ة → ه
  7. hamza:  ء removed, ؤ → و, ئ → ي
  8. Arabic-Indic / Persian digits → Western
  9. punctuation → space (Arabic ، ؛ ؟ ٪ and Latin)
 10. whitespace collapse
 11. optional: variant table (word-level, after everything else)

Not done on purpose: no de-duplication of repeated phrases, no lowercasing
(Arabic has no case; Latin islands keep case so English errors still count),
no number-word expansion (that lives in ``numbers.py`` for the numeral probe).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_TASHKEEL = "".join(chr(c) for c in list(range(0x064B, 0x0660)) + [0x0670])
_TATWEEL = "ـ"
_ALEF_VARIANTS = "أإآٱ"
_YA_VARIANT = "ى"
_TA_MARBUTA = "ة"
_HAMZA = "ء"

_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

_PUNCT_RE = re.compile(
    r"[،؛؟٪٫٬٭«»"
    r"\.,;:!\?\"'`’“”\(\)\[\]\{\}\-—–_/\\|<>~@#\$%\^&\*\+=]"
)
_WS_RE = re.compile(r"\s+")

_TASHKEEL_TABLE = {ord(c): None for c in _TASHKEEL}
_ALEF_TABLE = {ord(c): "ا" for c in _ALEF_VARIANTS}


@lru_cache(maxsize=1)
def _load_variants(path: str | None) -> dict[str, str]:
    """Word → canonical word. Keys and values are stored already-normalized."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    import yaml

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    table: dict[str, str] = {}
    for canonical, variants in (raw.get("variants") or {}).items():
        canon = _core(canonical)
        for v in variants or []:
            table[_core(v)] = canon
    return table


def _core(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text)
    s = s.translate(_TASHKEEL_TABLE)
    s = s.replace(_TATWEEL, "")
    s = s.translate(_ALEF_TABLE)
    s = s.replace(_YA_VARIANT, "ي")
    s = s.replace(_TA_MARBUTA, "ه")
    s = s.replace(_HAMZA, "").replace("ؤ", "و").replace("ئ", "ي")
    s = s.translate(_DIGITS)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_tolerant(text: str, *, variants_path: str | None = None) -> str:
    """Normalize for word-level scoring. ``variants_path`` enables the table."""
    s = _core(text)
    table = _load_variants(variants_path)
    if table and s:
        s = " ".join(table.get(w, w) for w in s.split())
    return s
