# Methodology (v0)

This page is the replication record. Everything needed to reproduce every number on the board is listed here or linked from here.

## What is measured

Zero-shot transcription of Arabic test utterances by systems "as released": open weights run with the decode parameters in each `results/<model>/run.json`, vendor APIs called with the fixed language hint recorded there. No fine-tuning, no prompt engineering, no per-dataset tuning.

## Test sets

The same six public test sets as Elm's Open Universal Arabic ASR Leaderboard, with the same utterance membership (reconstructed from Elm's published manifests, see `data/elm/`): SADA, Common Voice 18 (ar), MASC clean, MASC noisy, MGB-2, Casablanca. Audio is re-cut to 16 kHz mono PCM16; every file's SHA-256 is in the manifest. Where our reconstruction differs from Elm's membership the count is printed in the dataset build log and on the dataset card; we never claim identity without the check.

Two slices:

- **full**: every utterance. Open models only (cost).
- **common slice** (`api` in the code): a fixed, seeded, dialect-stratified sample (about 1,000 utterances per dataset, 30-utterance floor per dialect), kept for systems that cannot be run on everything, such as paid APIs in a later version. In v0 every open model ran on the full sets and the slice is scored from the same outputs.

The manifest hash (`data/manifests/MANIFEST_SHA256`) covers utterance ids, references, dialect labels, slice membership and audio hashes. A result is shown only if its `run.json` carries the current hash.

## Normalization (two published normalizers)

- `elm`: a line-for-line port of Elm's `normalize_arabic_text`, tested for string equality against the vendored original. Use this column to compare with Elm's board.
- `tolerant`: NFC, tashkeel, tatweel, alef/alef-maksura/ta-marbuta/hamza folding, digit unification, punctuation to space, whitespace, then an optional published spelling-variant table (`data/variants.yaml`).

Both are applied identically to reference and hypothesis. Code: `tahqiq/normalize/`.

## Metrics

- WER and CER are corpus-level (total edits / total reference tokens), as in Elm and NeMo.
- Every number carries a 95% utterance-level percentile bootstrap interval (2,000 resamples, seed 0).
- "vs baseline" is the paired bootstrap of WER(system) minus WER(Whisper large-v3) on shared utterances, plus McNemar's test on exact-match correctness.
- Numeral probe: on utterances whose reference contains digits or an unambiguous spoken number word (`tahqiq/normalize/numbers.py`), 1 if reference and hypothesis give the same digit string with number words mapped to digits on both sides, else 0. Wilson interval.
- Entity probe: reference entities are tagged once with CAMeL-Lab's `bert-base-arabic-camelbert-mix-ner` (`data/probes/entities.jsonl`, published), filtered to PERS/LOC/ORG with tagger score >= 0.8, no sub-word fragments, >= 3 characters, religious formulas removed. Score = share of reference entities whose tolerant-normalized form appears in the tolerant-normalized hypothesis. Wilson interval. Both probes are computed at scoring time from the reference text, so they never touch the manifest hash.
- Dialect columns pool utterances by the dialect label the source dataset provides (mapping in `tahqiq/manifest.py`); rows with fewer than 20 utterances are not shown.
- Coverage: the number of utterances a system returned output for is shown next to its score. Errors are not silently dropped from the denominator without being counted.

## Compute

v0 runs were made on shared NVIDIA H20 GPUs; wall times per utterance are recorded in every results line but are not a board column, because the GPUs were shared with other jobs and the numbers would not be comparable. A controlled speed column is planned.

## Scope and vendor policy

Systems on the board in v0 are open-weight models. A later version may add the speech APIs of global vendors. Vendors domiciled in the GCC appear only after they opt in in writing. Cohere's hosted API is excluded by its terms; the open-weight Cohere Transcribe Arabic checkpoint is included. All text on the site is numeric; the board makes no claims beyond the measured values.

## Reproduce one row

```
git clone https://github.com/h9-tec/tahqiq && cd tahqiq
uv sync --extra data --extra gpu
uv run python scripts/prepare_data.py casablanca            # needs the dataset's own access terms accepted
uv run python -m tahqiq.runner --model whisper_large_v3_turbo --dataset casablanca --slice api
uv run python scripts/build.py build
```
