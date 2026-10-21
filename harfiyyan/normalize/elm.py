"""Line-for-line port of Elm's leaderboard normalizer.

Source: ``eval.py`` in
https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard
(function ``normalize_arabic_text``). The original is vendored unchanged under
``tests/oracle/elm_eval_original.py`` and the test suite asserts exact string
equality between the two on every reference in every manifest.

Order of operations is part of the contract. Do not "improve" this file; put
improvements in ``tolerant.py``.

Known properties (documented, not bugs):
- punctuation is deleted, not replaced by a space ("كلمة،كلمة" becomes one token)
- no NFC normalization
- ة / ه and ى / ي are NOT folded
- Latin letters are kept, so code-switched English counts toward WER
- a standalone "و" is glued to the following word
"""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟]')
_DIACRITICS_RE = re.compile(r"[ً-ْ]")
_WAW_RE = re.compile(r"(^|\s)و\s+")
_WS_RE = re.compile(r"\s+")

_EASTERN_TO_WESTERN = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}


def normalize_elm(text: str) -> str:
    """Elm's ``normalize_arabic_text``, step for step."""
    text = _PUNCT_RE.sub("", text)
    text = _DIACRITICS_RE.sub("", text)

    text = re.sub("پ", "ب", text)
    text = re.sub("ڤ", "ف", text)
    text = re.sub(r"[آ]", "ا", text)
    text = re.sub(r"[أإ]", "ا", text)
    text = re.sub(r"[ؤ]", "و", text)
    text = re.sub(r"[ئ]", "ي", text)
    text = re.sub(r"[ء]", "", text)

    for eastern, western in _EASTERN_TO_WESTERN.items():
        text = text.replace(eastern, western)

    text = _WAW_RE.sub(r"\1و", text)
    text = _WS_RE.sub(" ", text).strip()
    return text
