"""Tag named entities in every reference once; write data/probes/entities.jsonl.

Uses CAMeL-Lab/bert-base-arabic-camelbert-mix-ner through a token-classification
pipeline (PER, LOC, ORG, MISC). The output is a side file keyed by utterance id,
so the manifest hash does not depend on the tagger. The entity probe then asks,
per reference entity, whether its tolerant-normalized form appears in the
tolerant-normalized hypothesis.

    uv run python scripts/make_entities.py            # all manifests
    uv run python scripts/make_entities.py --limit 50 # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tahqiq.manifest import load_manifest, manifest_paths  # noqa: E402

MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner"
OUT = ROOT / "data" / "probes" / "entities.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import torch
    from transformers import pipeline

    ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple",
                   device=0 if torch.cuda.is_available() else -1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_utts = n_ents = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for p in manifest_paths(ROOT / "data" / "manifests"):
            utts = load_manifest(p)
            if args.limit:
                utts = utts[: args.limit]
            for i in range(0, len(utts), args.batch):
                chunk = utts[i: i + args.batch]
                outs = ner([u.reference for u in chunk], batch_size=args.batch)
                for u, ents in zip(chunk, outs):
                    spans = [{"text": e["word"].strip(), "type": e["entity_group"], "score": round(float(e["score"]), 3)}
                             for e in ents if e["score"] >= 0.6 and len(e["word"].strip()) >= 2]
                    f.write(json.dumps({"id": u.id, "dataset": u.dataset, "entities": spans}, ensure_ascii=False) + "\n")
                    n_utts += 1
                    n_ents += len(spans)
            print(f"{p.stem}: done", flush=True)
    print(f"{n_utts} utterances, {n_ents} entities -> {OUT}")


if __name__ == "__main__":
    main()
