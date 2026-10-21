# Methodology (v0, draft)

This page is the replication record. Everything needed to reproduce every number on the board is listed here or linked from here.

## What is measured

Zero-shot transcription of Arabic test utterances by systems "as released": open weights run with the decode parameters in each `results/<model>/run.json`, vendor APIs called with the fixed language hint recorded there. No fine-tuning, no prompt engineering, no per-dataset tuning.

## Test sets

The same six public test sets as Elm's Open Universal Arabic ASR Leaderboard, with the same utterance membership (reconstructed from Elm's published manifests, see `data/elm/`): SADA, Common Voice 18 (ar), MASC clean, MASC noisy, MGB-2, Casablanca. Audio is re-cut to 16 kHz mono PCM16; every file's SHA-256 is in the manifest. Where our reconstruction differs from Elm's membership the count is printed in the dataset build log and on the dataset card; we never claim identity without the check.

Two slices:

- **full**: every utterance. Open models only (cost).
- **api-slice**: a fixed, seeded, dialect-stratified sample (about 1,000 utterances per dataset, 30-utterance floor per dialect). Every system on the board ran on exactly these utterances, so the headline table compares identical audio.

The manifest hash (`data/manifests/MANIFEST_SHA256`) covers utterance ids, references, dialect labels, slice membership and audio hashes. A result is shown only if its `run.json` carries the current hash.

## Normalization (two published normalizers)

- `elm`: a line-for-line port of Elm's `normalize_arabic_text`, tested for string equality against the vendored original. Use this column to compare with Elm's board.
- `tolerant`: NFC, tashkeel, tatweel, alef/alef-maksura/ta-marbuta/hamza folding, digit unification, punctuation to space, whitespace, then an optional published spelling-variant table (`data/variants.yaml`).

Both are applied identically to reference and hypothesis. Code: `harfiyyan/normalize/`.

## Metrics

- WER and CER are corpus-level (total edits / total reference tokens), as in Elm and NeMo.
- Every number carries a 95% utterance-level percentile bootstrap interval (2,000 resamples, seed 0).
- "vs baseline" is the paired bootstrap of WER(system) minus WER(Whisper large-v3) on shared utterances, plus McNemar's test on exact-match correctness.
- Numeral probe: on utterances whose reference contains written digits, 1 if the hypothesis contains the same digit sequence after normalization, else 0. Wilson interval.
- Entity probe (M3): span containment of reference named entities after tolerant normalization.
- Dialect columns pool utterances by the dialect label the source dataset provides (mapping in `harfiyyan/manifest.py`); rows with fewer than 20 utterances are not shown.
- Coverage: the number of utterances a system returned output for is shown next to its score. Errors are not silently dropped from the denominator without being counted.

## Speed and price

Batch speed factor = audio seconds / wall seconds on a fixed 50-utterance timing slice, concurrency 1, median of 3 passes; open models measured warm on one RTX 5080 laptop GPU, APIs measured from the same machine including upload and network. This is not streaming latency. Price per 1,000 minutes is the vendor's published pay-as-you-go rate on the date in `data/pricing.yaml`.

## Scope and vendor policy

Systems on the board are open-weight models and the speech APIs of global vendors. Vendors domiciled in the GCC appear only after they opt in in writing. Cohere's hosted API is excluded by its terms; the open-weight Cohere Transcribe Arabic checkpoint is included. All text on the site is numeric; the board makes no claims beyond the measured values.

## Reproduce one row

```
git clone https://github.com/h9-tec/harfiyyan && cd harfiyyan
uv sync --extra data --extra gpu
uv run python scripts/prepare_data.py casablanca            # needs the dataset's own access terms accepted
uv run python -m harfiyyan.runner --model whisper_large_v3_turbo --dataset casablanca --slice api
uv run python scripts/build.py build
```
