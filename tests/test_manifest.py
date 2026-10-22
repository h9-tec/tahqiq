import json
from pathlib import Path

from tahqiq.manifest import (
    Utterance,
    compute_manifest_hash,
    load_manifest,
    make_id,
    map_dialect,
    write_manifest,
)
from tahqiq.slices import assign_slices


def _utt(i: int, dialect: str = "gulf", dur: float = 5.0) -> Utterance:
    return Utterance(
        id=make_id("ds", str(i)), dataset="ds", split="test", source_id=str(i),
        audio_rel=f"ds/{i}.wav", audio_sha256="x" * 64, duration_s=dur,
        reference="كلمة", dialect_raw=dialect, dialect=map_dialect(dialect),
    )


def test_map_dialect():
    assert map_dialect("UAE") == "gulf"
    assert map_dialect("Algeria") == "maghrebi"
    assert map_dialect("Egypt") == "egyptian"
    assert map_dialect(None) == "unknown"
    assert map_dialect("Klingon") == "unknown"


def test_roundtrip_and_hash_key_order_independent(tmp_path: Path):
    utts = [_utt(i) for i in range(5)]
    write_manifest(tmp_path / "ds.jsonl", utts)
    back = load_manifest(tmp_path / "ds.jsonl")
    assert back == utts
    h1 = compute_manifest_hash(tmp_path)
    # Rewrite with shuffled key order; hash must not change.
    lines = []
    for line in (tmp_path / "ds.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        lines.append(json.dumps(dict(reversed(list(d.items()))), ensure_ascii=False))
    (tmp_path / "ds.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert compute_manifest_hash(tmp_path) == h1
    # Changing an audio hash changes the manifest hash.
    utts[0].audio_sha256 = "y" * 64
    write_manifest(tmp_path / "ds.jsonl", utts)
    assert compute_manifest_hash(tmp_path) != h1


def test_slices_stratified_and_deterministic():
    utts = [_utt(i, "gulf") for i in range(600)] + [_utt(1000 + i, "egyptian") for i in range(400)]
    utts += [_utt(5000 + i, "yemeni") for i in range(40)]
    assign_slices(utts, n_api=200, n_timing=10, seed=0)
    api = [u for u in utts if u.in_api_slice]
    by = {}
    for u in api:
        by[u.dialect] = by.get(u.dialect, 0) + 1
    assert by["yemeni"] == 30            # floor applies
    assert 100 <= by["gulf"] <= 130 and 60 <= by["egyptian"] <= 90
    assert sum(1 for u in utts if u.in_timing_slice) == 10
    first = {u.id for u in api}
    assign_slices(utts, n_api=200, n_timing=10, seed=0)
    assert {u.id for u in utts if u.in_api_slice} == first
