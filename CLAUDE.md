# Tahqiq (تحقيق): working rules

- Numbers on the board come only from `results/` via `scripts/build.py`. Never hand-edit a score.
- Two normalizers, both published: `elm` (comparability with Elm's leaderboard, do not change) and `tolerant`. New folding goes in `tolerant.py` + `data/variants.yaml` with a test.
- Every result carries the manifest hash in `run.json`; results with a stale hash are not shown.
- Vendor policy: global vendors and open weights only. GCC-domiciled vendors appear only with signed opt-in. Cohere via open weights only. Numeric register in all public text; no editorial claims about any vendor.
- Keys come from `.env` (gitignored) through the environment. Never commit audio or keys.
- Tests: `uv run pytest` (CI-safe). `-m gpu` / `-m api` tests need weights or keys and run locally.
- Commits: plain messages, no attribution trailers.
