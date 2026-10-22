"""Transcript-convention fingerprint: does a system write like a corpus's annotators on that corpus only?

Every corpus has orthographic habits (MASC captions drop hamza and ta-marbuta; Common Voice keeps
hamza and punctuation; MGB-2 and SADA have their own). A system trained on a corpus's transcripts
tends to reproduce those habits *on that corpus* and not elsewhere. We measure, per utterance, a
small vector of convention features on the raw (un-normalized) text, then compare how closely a
system's hypotheses track the references' conventions on each set.

Features (rates per character or per word, raw text):
  hamza_forms   share of alef letters written with hamza (أإآ) among all alef-type letters
  ta_marbuta    share of word-final ة among words ending in ة/ه
  alef_maksura  share of word-final ى among words ending in ى/ي
  diacritics    tashkeel marks per character
  punctuation   punctuation marks per word
  digits        share of digit tokens among numeric tokens (digits vs number words)
  latin         share of Latin-script characters

Score for (system m, corpus d): distance between the system's mean feature vector on d and the
references' mean vector on d, relative to the system's distance on the other corpora. A system
whose distance on d is far smaller than its own typical distance (robust z ≤ -2.5) and smaller
than peers' is flagged "convention match". It is evidence of training on the corpus's transcripts,
not proof: a model can also just share a normalization convention with one corpus.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

_TASHKEEL = re.compile(r"[ً-ْٰ]")
_PUNCT = re.compile(r"[،؛؟!\.,;:\?\"'«»\(\)\[\]]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT_TOKEN = re.compile(r"^[0-9٠-٩]+$")
_NUMBER_WORDS = {"واحد", "اثنين", "اتنين", "ثلاثة", "ثلاث", "تلاتة", "أربعة", "اربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة", "عشرين", "ثلاثين", "أربعين", "خمسين", "مائة", "مية", "ألف", "الف"}
FEATURES = ("hamza_forms", "ta_marbuta", "alef_maksura", "diacritics", "punctuation", "digits", "latin")


def features(text: str) -> dict[str, float] | None:
    t = text or ""
    words = [w for w in (_PUNCT.sub("", x) for x in t.split()) if w]   # endings are judged without trailing punctuation
    if not words:
        return None
    chars = len(t)
    alef_all = sum(t.count(c) for c in "اأإآ")
    alef_h = sum(t.count(c) for c in "أإآ")
    end_th = [w for w in words if w[-1] in "ةه"]
    end_ty = [w for w in words if w[-1] in "ىي"]
    num_tokens = [w for w in words if _DIGIT_TOKEN.match(w) or w in _NUMBER_WORDS]
    return {
        "hamza_forms": alef_h / alef_all if alef_all else 0.0,
        "ta_marbuta": sum(1 for w in end_th if w[-1] == "ة") / len(end_th) if end_th else 0.0,
        "alef_maksura": sum(1 for w in end_ty if w[-1] == "ى") / len(end_ty) if end_ty else 0.0,
        "diacritics": len(_TASHKEEL.findall(t)) / chars,
        "punctuation": len(_PUNCT.findall(t)) / len(words),
        "digits": sum(1 for w in num_tokens if _DIGIT_TOKEN.match(w)) / len(num_tokens) if num_tokens else 0.0,
        "latin": len(_LATIN.findall(t)) / chars,
    }


def mean_vector(texts: list[str]) -> dict[str, float] | None:
    acc: dict[str, float] = defaultdict(float)
    n = 0
    for s in texts:
        f = features(s)
        if f is None:
            continue
        n += 1
        for k, v in f.items():
            acc[k] += v
    if n == 0:
        return None
    return {k: acc[k] / n for k in FEATURES}


def distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in FEATURES))


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def fingerprint_report(ref_vectors: dict[str, dict[str, float]], hyp_vectors: dict[str, dict[str, dict[str, float]]],
                       *, z_t: float = -2.5, min_sets: int = 4) -> dict:
    """ref_vectors: dataset → mean features of references.
    hyp_vectors: model → dataset → mean features of that model's hypotheses.
    Returns per-model per-dataset distances and flags."""
    out: dict = {"distances": {}, "flags": {}}
    for m, per in hyp_vectors.items():
        dists = {d: distance(v, ref_vectors[d]) for d, v in per.items() if d in ref_vectors}
        out["distances"][m] = {d: round(x, 4) for d, x in dists.items()}
        if len(dists) < min_sets:
            continue
        logs = {d: math.log(max(x, 1e-4)) for d, x in dists.items()}
        med = _median(list(logs.values()))
        mad = _median([abs(v - med) for v in logs.values()]) or 1e-6
        for d, lv in logs.items():
            z = 0.6745 * (lv - med) / mad
            if z <= z_t:
                out["flags"].setdefault(m, {})[d] = {"z": round(z, 2), "distance": round(dists[d], 4), "typical": round(math.exp(med), 4)}
    return out
