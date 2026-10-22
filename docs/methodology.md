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

## Production profile

Two groups of columns, kept apart on the board.

**Measured (scripts/timing.py).** A fixed timing slice (`in_timing_slice` in the manifests, about 60 utterances of 3 to 15 s drawn from every set), one warm-up pass, then three timed passes at batch size 1, concurrency 1, on one NVIDIA H20. RTFx = total audio seconds / total wall seconds, median of the three passes. Latency = wall seconds per utterance, p50 and p95 over all timed utterances (this is offline, per-file latency, not streaming time-to-first-word). Peak VRAM = `torch.cuda.max_memory_allocated` after the passes. Load time = seconds from process start to a ready model, weights already on local disk. The box is shared; the number of other processes on the GPU during the measurement is stored in `timing.json` and shown as a flag when it is not zero.

**Declared (data/models.yaml).** Streaming (native / chunked / no), timestamps (word / segment / char / no), maximum input per call, languages, Arabic-only, with the model card or paper cited per entry; license, release date, weight size and parameter count come from the Hub API. Community submissions start as "unknown" until a maintainer fills the entry from the card. Declared fields are never measured by us and are labelled as declared.

## Compute

v0 runs were made on shared NVIDIA H20 GPUs; wall times per utterance are recorded in every results line but are not a board column, because the GPUs were shared with other jobs and the numbers would not be comparable. A controlled speed column is planned.

## Contamination

Two signals, shown per (system, test set) cell on the board, defined in `tahqiq/contamination.py`:

- **Declared (D).** `data/contamination.yaml` lists systems whose model card or paper names a test corpus (SADA, MGB-2, Common Voice, MASC, Casablanca) in their training data, with the source quoted. Such cells are marked in-domain and excluded from that system's average by default (a toggle shows the unfiltered average). Common Voice re-splits between releases, so training on any CV release counts.
- **Anomalous advantage (A).** With at least six systems on the board, log WER is fit with an additive median polish (system effect + test-set effect). A cell whose residual is ≤ −0.30 (about 26% lower WER than the fit predicts) and whose robust z-score is ≤ −2.5 is flagged. This is the signature of a test-set leak; it is not proof, and the thresholds and the fit are published so anyone can recompute them from `board.json`.

- **Convention fingerprint (F).** Each corpus has transcription habits (MASC captions carry no hamza and no ta-marbuta; Common Voice keeps diacritics; MGB-2 writes ta-marbuta 61% of the time). `tahqiq/fingerprint.py` measures seven such rates on raw text. A system whose hypotheses on one corpus track that corpus's reference conventions far more closely than its hypotheses track any other corpus's conventions (robust z ≤ −2.5 across its own sets) is flagged. Reference vectors are in `data/probes/ref_conventions.json`.
- **Dev-vs-test gap (G).** About 500 utterances per corpus are drawn deterministically from each corpus's public validation split (`scripts/prepare_dev.py`). A clean system scores about the same on dev and test; a system that saw the test split does markedly better on test. Flag: log WER(test, common slice) − log WER(dev) ≤ −0.25 and robust z across systems on that corpus ≤ −2.5.
- **Perturbation gap (P).** The common slice is re-run on four word-preserving variants (`scripts/make_perturbed.py`: speed 0.94 and 1.06, 20 dB pink noise, telephone band). Memorized audio or transcripts lose their advantage under perturbation, so a contaminated system degrades more on that corpus than it does on its other corpora and than peers do. Flag: mean log-WER delta on the corpus exceeds the system's own median delta by ≥ 0.15 with robust z ≥ 2.5 across systems.
- **Likelihood probe (L).** For decoders that can score a transcript (Whisper, Cohere Transcribe), the teacher-forced NLL of the reference is compared with the NLL of the system's own greedy output on each common-slice utterance. Systems normally prefer their own output; memorized references flip that. Flag: the corpus's reference-preferred rate exceeds the system's own median by ≥ 0.15 with robust z ≤ −2.5.
- **Private matched set (v1).** The only signal that does not depend on what a model declares: a freshly recorded slice matched to each public set, with the protocol fixed in advance ([docs/private_set_protocol.md](private_set_protocol.md)). The published number is log WER(public) − log WER(private) per corpus.

None of the automatic signals proves contamination on its own; each is published with its threshold and inputs so it can be recomputed, and a cell can carry several markers.

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
