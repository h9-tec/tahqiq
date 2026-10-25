"""Recompute the production profile (Hub facts + declared fields from data/models.yaml) for every published
model and push the changed scores files in one commit, then rebuild board.json. Scores are not touched.

    .venv/bin/python scripts/refresh_profiles.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from huggingface_hub import CommitOperationAdd  # noqa: E402

from tahqiq import hub  # noqa: E402
from tahqiq.profile import profile  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ops, changed = [], []
    for doc in hub.load_all_scores(force=True):
        slug = doc["model_id"]
        new = profile(slug, doc.get("hf_id"))
        if "hub_error" in new and doc.get("profile"):
            print(f"  keep {slug}: {new['hub_error']}")
            continue
        old = doc.get("profile") or {}
        diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k) and k != "downloads"}
        if not diff:
            continue
        changed.append(slug)
        print(f"  {slug}: " + ", ".join(f"{k} {o!r}->{n!r}" for k, (o, n) in sorted(diff.items()))[:300])
        doc["profile"] = new
        ops.append(CommitOperationAdd(path_in_repo=f"scores/{slug}.json",
                                      path_or_fileobj=json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8")))
    print(f"{len(changed)} profiles changed")
    if a.dry_run or not ops:
        return
    hub._api().create_commit(repo_id=hub.RESULTS_REPO, repo_type="dataset", operations=ops,
                             commit_message=f"profiles: refresh {len(ops)} models (weights size, declared fields)")
    b = hub.rebuild_board()
    print("board rebuilt", b["generated_at"], len(b["models"]), "models")


if __name__ == "__main__":
    main()
