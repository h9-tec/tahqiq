"""Manifest records, loading, and the published manifest hash.

One JSONL file per dataset under ``data/manifests/``. Every line is one test
utterance. The hash over all manifests (``data/manifests/MANIFEST_SHA256``) is
printed on the site and stored in every ``run.json``; a result is only shown
on the board if its hash equals the current one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable

DIALECTS = (
    "msa", "gulf", "najdi", "hijazi", "egyptian", "levantine",
    "maghrebi", "yemeni", "iraqi", "sudanese", "unknown",
)

# Country / raw-label → taxonomy. Keys are lower-case.
DIALECT_MAP = {
    "algeria": "maghrebi", "morocco": "maghrebi", "mauritania": "maghrebi", "tunisia": "maghrebi",
    "libya": "maghrebi",
    "egypt": "egyptian", "egyptian": "egyptian",
    "jordan": "levantine", "palestine": "levantine", "lebanon": "levantine", "syria": "levantine",
    "levantine": "levantine",
    "uae": "gulf", "kuwait": "gulf", "bahrain": "gulf", "qatar": "gulf", "oman": "gulf",
    "khaliji": "gulf", "gulf": "gulf",
    "najdi": "najdi", "hijazi": "hijazi", "saudi": "gulf",
    "yemen": "yemeni", "yemeni": "yemeni",
    "iraq": "iraqi", "iraqi": "iraqi",
    "sudan": "sudanese", "sudanese": "sudanese",
    "msa": "msa", "modern standard arabic": "msa", "modernstandardarabic": "msa",
    # SADA labels
    "shamali": "gulf",  # northern Saudi
    "more than 1 speaker اكثر من متحدث": "unknown", "notapplicable": "unknown",
}


def map_dialect(raw: str | None) -> str:
    if not raw:
        return "unknown"
    return DIALECT_MAP.get(raw.strip().lower(), "unknown")


@dataclass
class Utterance:
    id: str
    dataset: str
    split: str
    source_id: str
    audio_rel: str
    audio_sha256: str
    duration_s: float
    reference: str
    dialect_raw: str = ""
    dialect: str = "unknown"
    has_number: bool = False
    number_digits: str = ""
    has_entity: bool = False
    entities: list[str] = field(default_factory=list)
    in_api_slice: bool = False
    in_timing_slice: bool = False
    elm_member: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Utterance":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


def make_id(dataset: str, source_id: str) -> str:
    return hashlib.sha256(f"{dataset}|{source_id}".encode("utf-8")).hexdigest()[:16]


def canonical_line(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_manifest(path: Path, utts: Iterable[Utterance]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u in utts:
            f.write(canonical_line(u.to_dict()) + "\n")


def load_manifest(path: Path) -> list[Utterance]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Utterance.from_dict(json.loads(line)))
    return out


def manifest_paths(manifests_dir: Path) -> list[Path]:
    return sorted(p for p in manifests_dir.glob("*.jsonl"))


def compute_manifest_hash(manifests_dir: Path) -> str:
    """sha256 over canonical lines of every manifest, in file-name order.

    Lines are re-canonicalized, so key order on disk does not matter; the
    audio hashes are inside the lines, so changed audio changes the hash.
    """
    h = hashlib.sha256()
    for p in manifest_paths(manifests_dir):
        h.update(p.name.encode("utf-8"))
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    h.update(canonical_line(json.loads(line)).encode("utf-8"))
                    h.update(b"\n")
    return h.hexdigest()


def write_manifest_hash(manifests_dir: Path) -> str:
    digest = compute_manifest_hash(manifests_dir)
    (manifests_dir / "MANIFEST_SHA256").write_text(digest + "\n", encoding="utf-8")
    return digest


def read_manifest_hash(manifests_dir: Path) -> str | None:
    p = manifests_dir / "MANIFEST_SHA256"
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
