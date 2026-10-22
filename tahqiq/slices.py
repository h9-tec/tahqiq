"""Slice assignment: which utterances go to the api-slice and the timing-slice.

Deterministic (seed 0), stratified by ``dialect`` so every dialect present in
a dataset is represented in proportion, with a floor so small dialects are not
starved.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .manifest import Utterance


def stratified_sample(utts: list[Utterance], n: int, *, seed: int = 0, floor: int = 30) -> set[str]:
    """Return ids of ~n utterances, proportional by dialect with a per-dialect floor."""
    if n >= len(utts):
        return {u.id for u in utts}
    rng = random.Random(seed)
    groups: dict[str, list[Utterance]] = defaultdict(list)
    for u in utts:
        groups[u.dialect].append(u)
    total = len(utts)
    chosen: set[str] = set()
    for dialect, items in sorted(groups.items()):
        k = max(min(floor, len(items)), round(n * len(items) / total))
        k = min(k, len(items))
        items_sorted = sorted(items, key=lambda u: u.id)
        chosen.update(u.id for u in rng.sample(items_sorted, k))
    return chosen


def timing_sample(utts: list[Utterance], n: int = 50, *, seed: int = 0, lo: float = 3.0, hi: float = 15.0) -> set[str]:
    rng = random.Random(seed)
    pool = sorted((u for u in utts if lo <= u.duration_s <= hi), key=lambda u: u.id)
    return {u.id for u in rng.sample(pool, min(n, len(pool)))}


def assign_slices(utts: list[Utterance], *, n_api: int, n_timing: int, seed: int = 0) -> None:
    api = stratified_sample(utts, n_api, seed=seed)
    timing = timing_sample(utts, n_timing, seed=seed)
    for u in utts:
        u.in_api_slice = u.id in api
        u.in_timing_slice = u.id in timing
