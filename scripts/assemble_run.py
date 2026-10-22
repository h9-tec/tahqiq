"""Assemble results/<slug>/run.json from partial (helper) runs made on several machines.

    uv run python scripts/assemble_run.py whisper_large_v3

Takes the unit runs under results/<slug>/runs/*.json (each lists the datasets it covered),
checks every manifest dataset has a complete cache, and writes one run.json carrying the
manifest hash, the union of datasets, the environment of the first unit, and summed totals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tahqiq import io as hio  # noqa: E402
from tahqiq.manifest import load_manifest, manifest_paths, read_manifest_hash  # noqa: E402

MANIFESTS = ROOT / "data" / "manifests"
RESULTS = ROOT / "results"


def assemble(slug: str) -> dict:
    d = RESULTS / slug
    units = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((d / "runs").glob("*.json"))]
    if not units:
        raise SystemExit(f"{slug}: no unit runs under {d/'runs'}")
    current = read_manifest_hash(MANIFESTS)
    bad = [u["run_id"] for u in units if u.get("manifest_sha256") != current]
    if bad:
        raise SystemExit(f"{slug}: units with a stale manifest hash: {bad}")
    datasets = [p.stem for p in manifest_paths(MANIFESTS)]
    coverage = {}
    for ds in datasets:
        utts = load_manifest(MANIFESTS / f"{ds}.jsonl")
        preds = hio.load_predictions(RESULTS, slug, ds)
        ok = sum(1 for u in utts if u.id in preds and not preds[u.id].error)
        coverage[ds] = (ok, len(utts))
    missing = {ds: v for ds, v in coverage.items() if v[0] < v[1]}
    base = units[0]
    totals = {k: sum(u["totals"].get(k, 0) for u in units) for k in ("done", "errors", "skipped", "audio_s", "wall_s")}
    run = {**base, "run_id": "+".join(u["run_id"] for u in units), "datasets": datasets, "slice": "full",
           "started": min(u["started"] for u in units), "finished": max(u["finished"] for u in units),
           "totals": totals, "assembled_from": [u["run_id"] for u in units],
           "machines": sorted({u.get("env", {}).get("machine", "?") for u in units})}
    (d / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{slug}: coverage " + ", ".join(f"{ds} {a}/{b}" for ds, (a, b) in coverage.items()) + (f"  MISSING {missing}" if missing else "  complete"))
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    assemble(ap.parse_args().slug)
