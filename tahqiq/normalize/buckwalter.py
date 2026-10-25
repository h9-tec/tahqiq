"""Buckwalter transliteration -> Arabic script (Tim Buckwalter's table, with the XML-safe extensions).

Some early wav2vec2 Arabic fine-tunes (trained on Common Voice / Arabic Speech Corpus in Buckwalter) emit
Buckwalter strings; their cards convert back with ``lang_trans.arabic.buckwalter.untrans``. Their vocabularies
use "/" as the word delimiter, since "|" is a letter (alef madda) in Buckwalter.
"""

from __future__ import annotations

import re

BW2AR = {
    "'": "ء", "|": "آ", ">": "أ", "&": "ؤ", "<": "إ", "}": "ئ", "A": "ا", "b": "ب", "p": "ة", "t": "ت", "v": "ث",
    "j": "ج", "H": "ح", "x": "خ", "d": "د", "*": "ذ", "r": "ر", "z": "ز", "s": "س", "$": "ش", "S": "ص", "D": "ض",
    "T": "ط", "Z": "ظ", "E": "ع", "g": "غ", "_": "ـ", "f": "ف", "q": "ق", "k": "ك", "l": "ل", "m": "م", "n": "ن",
    "h": "ه", "w": "و", "Y": "ى", "y": "ي", "F": "ً", "N": "ٌ", "K": "ٍ", "a": "َ", "u": "ُ", "i": "ِ", "~": "ّ",
    "o": "ْ", "`": "ٰ", "{": "ٱ", "P": "پ", "J": "چ", "V": "ڤ", "G": "گ",
}
_SIGNATURE = set("$*><&}|~")   # characters that only make sense as Buckwalter letters


def to_arabic(s: str, *, word_delimiter: str = "/") -> str:
    s = s.replace(word_delimiter, " ")
    return re.sub(r"\s+", " ", "".join(BW2AR.get(c, c) for c in s)).strip()


def looks_like_buckwalter_vocab(vocab: dict[str, int] | list[str]) -> bool:
    """A CTC vocabulary with no Arabic letters but Buckwalter-only symbols."""
    toks = list(vocab)
    if any("؀" <= c <= "ۿ" for t in toks for c in t):
        return False
    chars = {c for t in toks if len(t) == 1 for c in t}
    return len(_SIGNATURE & chars) >= 3 and {"A", "b", "t", "E", "H"} <= chars
