"""Text normalizers.

Two normalizers are published, by name, and every score on the board says
which one produced it:

- ``elm``: a line-for-line port of the Open Universal Arabic ASR Leaderboard's
  ``normalize_arabic_text`` (Elm Research Center). Use it when a number must
  be comparable with Elm's board.
- ``tolerant``: a stronger folding (NFC, tatweel, alef, alef-maksura, ta-marbuta,
  hamza, digits, punctuation) plus an optional spelling-variant table. Use it
  when the question is "did the model get the words" rather than "did it pick
  the same orthography as the transcriber".
"""

from .elm import normalize_elm
from .tolerant import normalize_tolerant

NORMALIZERS = {
    "elm": normalize_elm,
    "tolerant": normalize_tolerant,
}


def get(name: str):
    try:
        return NORMALIZERS[name]
    except KeyError:
        raise KeyError(f"unknown normalizer {name!r}; known: {sorted(NORMALIZERS)}") from None


__all__ = ["normalize_elm", "normalize_tolerant", "NORMALIZERS", "get"]
