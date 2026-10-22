"""Hugging Face Hub storage for the board: a public request queue and a public results store.

Repos (datasets):
  HeshamHaroon/tahqiq-requests   requests/<slug>.json   one file per submission, status machine
  HeshamHaroon/tahqiq-results    scores/<slug>.json     aggregated scores for the board
                                    results/<slug>/        per-utterance outputs + run.json

Status machine: PENDING → RUNNING → DONE | FAILED ; PENDING → NEEDS_REVIEW → (human sets) APPROVED → RUNNING.

This module depends only on huggingface_hub so the Space can import a copy of it.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REQUESTS_REPO = os.environ.get("TAHQIQ_REQUESTS_REPO", "HeshamHaroon/tahqiq-requests")
RESULTS_REPO = os.environ.get("TAHQIQ_RESULTS_REPO", "HeshamHaroon/tahqiq-results")
STATUSES = ("PENDING", "APPROVED", "RUNNING", "DONE", "FAILED", "NEEDS_REVIEW")


def _api() -> HfApi:
    return HfApi(token=os.environ.get("HF_TOKEN") or None)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug_for(hf_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", hf_id.strip()).strip("_")
    return s.replace("/", "__")[:120]


# ---------- requests ----------

def list_requests(*, force: bool = False) -> list[dict]:
    d = snapshot_download(REQUESTS_REPO, repo_type="dataset", allow_patterns=["requests/*.json"],
                          token=os.environ.get("HF_TOKEN") or None, force_download=force)
    out = []
    for p in sorted(Path(d, "requests").glob("*.json")) if Path(d, "requests").exists() else []:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def get_request(slug: str) -> dict | None:
    try:
        p = hf_hub_download(REQUESTS_REPO, f"requests/{slug}.json", repo_type="dataset",
                            token=os.environ.get("HF_TOKEN") or None, force_download=True)
    except Exception:
        return None
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_request(req: dict) -> None:
    req = dict(req)
    req["updated_at"] = now()
    _api().upload_file(
        path_or_fileobj=json.dumps(req, ensure_ascii=False, indent=1).encode("utf-8"),
        path_in_repo=f"requests/{req['slug']}.json",
        repo_id=REQUESTS_REPO, repo_type="dataset",
        commit_message=f"{req['slug']}: {req['status']}",
    )


def submit_request(hf_id: str, *, display_name: str = "", contact: str = "", notes: str = "", submitted_by: str = "") -> dict:
    hf_id = hf_id.strip().strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", hf_id):
        raise ValueError("model id must look like owner/name")
    info = _api().model_info(hf_id)  # raises if it does not exist or is not visible
    slug = slug_for(hf_id)
    existing = get_request(slug)
    if existing and existing["status"] in ("PENDING", "APPROVED", "RUNNING", "DONE"):
        return existing
    req = {
        "slug": slug, "hf_id": hf_id, "display_name": display_name.strip() or hf_id.split("/")[-1],
        "revision": getattr(info, "sha", None), "contact": contact.strip(), "notes": notes.strip()[:2000],
        "submitted_by": submitted_by, "submitted_at": now(), "status": "PENDING",
        "error": "", "adapter": "", "worker": "",
    }
    write_request(req)
    return req


def update_status(slug: str, status: str, **fields) -> dict:
    assert status in STATUSES, status
    req = get_request(slug)
    if req is None:
        raise KeyError(slug)
    req.update(fields)
    req["status"] = status
    write_request(req)
    return req


# ---------- results ----------

def upload_results(slug: str, local_dir: Path, scores: dict) -> None:
    api = _api()
    api.upload_folder(folder_path=str(local_dir), path_in_repo=f"results/{slug}", repo_id=RESULTS_REPO,
                      repo_type="dataset", commit_message=f"{slug}: per-utterance outputs",
                      ignore_patterns=["runs/*"])
    api.upload_file(path_or_fileobj=json.dumps(scores, ensure_ascii=False).encode("utf-8"),
                    path_in_repo=f"scores/{slug}.json", repo_id=RESULTS_REPO, repo_type="dataset",
                    commit_message=f"{slug}: scores")


def load_all_scores(*, force: bool = False) -> list[dict]:
    d = snapshot_download(RESULTS_REPO, repo_type="dataset", allow_patterns=["scores/*.json"],
                          token=os.environ.get("HF_TOKEN") or None, force_download=force)
    out = []
    for p in sorted(Path(d, "scores").glob("*.json")) if Path(d, "scores").exists() else []:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def results_url(slug: str) -> str:
    return f"https://huggingface.co/datasets/{RESULTS_REPO}/tree/main/results/{slug}"


def rebuild_board(*, force: bool = True) -> dict:
    """Aggregate scores/*.json + queue status into board.json at the results repo root (what the Space fetches)."""
    scores = load_all_scores(force=force)
    queue = [{k: r.get(k) for k in ("slug", "hf_id", "display_name", "status", "submitted_at", "updated_at", "error", "submitted_by")}
             for r in list_requests(force=force)]
    board = {"generated_at": now(), "results_repo": RESULTS_REPO, "requests_repo": REQUESTS_REPO,
             "models": scores, "queue": queue}
    _api().upload_file(path_or_fileobj=json.dumps(board, ensure_ascii=False).encode("utf-8"),
                       path_in_repo="board.json", repo_id=RESULTS_REPO, repo_type="dataset",
                       commit_message=f"board: {len(scores)} models, {len(queue)} requests")
    return board


# ---------- submissions through Hub discussions (works from a static Space with OAuth) ----------

SUBMIT_PREFIX = "submit:"


def poll_discussions() -> list[dict]:
    """Turn open 'submit: owner/model' discussions on the requests repo into request files.

    The Space creates such a discussion on the submitter's behalf (OAuth scope
    write-discussions). We answer in the thread and close it when the job ends.
    """
    api = _api()
    created = []
    for disc in api.get_repo_discussions(REQUESTS_REPO, repo_type="dataset", discussion_status="open"):
        if disc.is_pull_request or not disc.title.lower().startswith(SUBMIT_PREFIX):
            continue
        hf_id = disc.title[len(SUBMIT_PREFIX):].strip()
        slug = slug_for(hf_id)
        if get_request(slug):
            continue
        details = api.get_discussion_details(REQUESTS_REPO, disc.num, repo_type="dataset")
        body = ""
        for ev in details.events:
            if ev.type == "comment":
                body = getattr(ev, "content", "") or ""
                break
        meta: dict = {}
        try:
            meta = json.loads(body) if body.strip().startswith("{") else {}
        except Exception:
            meta = {}
        try:
            req = submit_request(hf_id, display_name=str(meta.get("display_name", "")), contact=str(meta.get("contact", "")),
                                 notes=str(meta.get("notes", ""))[:2000], submitted_by=disc.author or "")
        except Exception as e:  # noqa: BLE001
            api.comment_discussion(REQUESTS_REPO, disc.num, repo_type="dataset",
                                   comment=f"Could not queue `{hf_id}`: {type(e).__name__}: {e}")
            api.change_discussion_status(REQUESTS_REPO, disc.num, "closed", repo_type="dataset")
            continue
        req["discussion"] = disc.num
        write_request(req)
        api.comment_discussion(REQUESTS_REPO, disc.num, repo_type="dataset",
                               comment=f"Queued as `{slug}` (status PENDING). Results will appear at {results_url(slug)} when done.")
        created.append(req)
    return created


def notify(req: dict, text: str, *, close: bool = False) -> None:
    num = req.get("discussion")
    if not num:
        return
    api = _api()
    try:
        api.comment_discussion(REQUESTS_REPO, int(num), repo_type="dataset", comment=text)
        if close:
            api.change_discussion_status(REQUESTS_REPO, int(num), "closed", repo_type="dataset")
    except Exception:
        pass


def wait_for_hub(seconds: float) -> None:
    time.sleep(seconds)
