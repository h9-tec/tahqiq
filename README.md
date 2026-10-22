<p align="center"><img src="assets/banner.png" alt="Tahqiq" width="100%"></p>

<p align="center">
  <a href="https://huggingface.co/spaces/HeshamHaroon/tahqiq"><b>Live leaderboard</b></a> ·
  <a href="https://huggingface.co/spaces/HeshamHaroon/tahqiq#submit"><b>Submit a model</b></a> ·
  <a href="https://huggingface.co/datasets/HeshamHaroon/tahqiq-results">Results dataset</a> ·
  <a href="https://huggingface.co/datasets/HeshamHaroon/tahqiq-requests">Request queue</a> ·
  <a href="docs/methodology.md">Methodology</a>
</p>

# Tahqiq (تحقيق)

**An independent, reproducible leaderboard for Arabic speech recognition.** Open-weight models are run by one harness on the same six public test sets, scored the same way, and published with every per-utterance output. Anyone can submit a model from the Hugging Face Hub; it is evaluated on Hugging Face compute and appears on the board with its results thread.

*Tahqiq* is the most exact style of Quran recitation, where every letter is given its full right. In everyday Arabic the word means verification.

## Why another board

Arabic ASR numbers are mostly self-reported, scored with different normalizers on different slices of data, and published without uncertainty. Tahqiq fixes the parts that make numbers incomparable:

| | Typical leaderboard | Tahqiq |
|---|---|---|
| Who runs the evaluation | model authors | this harness, same code and settings for every system |
| Normalization | one, often undocumented | two published normalizers: `elm` (a verified 1:1 port of the Open Universal Arabic ASR Leaderboard's, for comparability) and `tolerant` (orthographic folding + spelling variants) |
| Uncertainty | none | 95% bootstrap interval on every number; paired delta and McNemar test against Whisper large-v3 |
| Beyond WER | none | numeral probe (digits and spoken numbers), named-entity probe (people, places, organizations) |
| Dialects | aggregate only | per-dialect columns wherever the test set carries labels |
| Production fit | none | measured RTFx, per-file latency p50/p95, peak VRAM, load time on a dedicated GPU; declared streaming, timestamps, max input, languages, license, size, with the card cited |
| Contamination | ignored | six signals per cell: declared overlap, cross-set anomaly, transcript-convention fingerprint, dev-vs-test gap, perturbation gap, likelihood probe; plus a private matched set protocol for v1 |
| Replication | partial | manifests with audio hashes, every hypothesis, `run.json` per system, deterministic slices |

## Test sets

The six public sets used by the Open Universal Arabic ASR Leaderboard, reconstructed utterance for utterance from its published manifests. Audio is obtained under each set's own terms and never redistributed; only outputs are published.

| Set | Utterances | Hours | Dialect labels |
|---|---:|---:|---|
| SADA | 6,193 | 10.8 | Najdi, Hijazi, Gulf, MSA, Egyptian |
| Casablanca | 6,818 | 7.9 | Algeria, Egypt, Jordan, Mauritania, Morocco, Palestine, UAE, Yemen |
| Common Voice 18 (ar) | 10,471 | 12.7 | mostly unlabelled |
| MASC clean | 8,612 | 10.5 | unlabelled |
| MASC noisy | 9,394 | 8.9 | unlabelled |
| MGB-2 | 5,365 | 9.6 | unlabelled, mostly MSA broadcast |

Every run also scores a fixed, seeded, dialect-stratified **common slice** of about 1,000 utterances per set. Community submissions are evaluated on that slice and are marked accordingly on the board.

## Submit a model

1. Open the [Space](https://huggingface.co/spaces/HeshamHaroon/tahqiq), tab **Submit a model**, sign in with Hugging Face (the only permission asked is to open a discussion in the request queue).
2. Enter the model id. Transformers-native ASR architectures (Whisper, wav2vec2/HuBERT CTC, Qwen3-ASR, Cohere Transcribe, Voxtral) run automatically. NeMo checkpoints and repositories that need `trust_remote_code` are parked for a maintainer.
3. The worker evaluates it on ZeroGPU (included with the maintainer's Hugging Face PRO plan, no pay-per-use) on the common slice of all six sets, scores it, measures timing, and replies in the request thread with the numbers and a link to every per-utterance output. Typical turnaround is under an hour. Rows evaluated on the common slice are labelled as such in every view.

## Leaderboard snapshot

<!-- tahqiq:table:start -->
| System | Avg WER | Casablanca | CV18 | MASC-c | MASC-n | MGB-2 | SADA | Numerals | Entities | Δ vs Whisper v3 |
|---|---|---|---|---|---|---|---|---|---|---|
| SeamlessM4T v2 large (arb) | 33.8 | 65.0 | 10.6 | 18.4 | 29.5 | 16.2 | 63.4 | 52% | 70% | base |
| Qwen3-ASR 0.6B | 40.9 | 70.2 | 27.6 | 28.4 | 41.7 | 25.1 | 52.6 | 66% | 54% | base |
| Whisper large-v3-turbo | 43.9 | 85.7 | 23.9 | 21.1 | 37.2 | 18.3 | 77.0 | 48% | 63% | base |
| MMS-1b-all (ara) | 53.6 | 86.0 | 26.0 | 36.5 | 56.6 | 39.0 | 77.4 | 35% | 44% | base |

WER % on the full test sets, `elm` normalizer, corpus-level, sorted by unweighted average over 6 sets. Confidence intervals, CER, the tolerant normalizer and per-dialect numbers are on the [live board](https://huggingface.co/spaces/HeshamHaroon/tahqiq). Snapshot of 4 systems, manifest `6ef6d50155f4`, 2026-10-22.
<!-- tahqiq:table:end -->

## Contamination

A public test set can be in any model's training data, declared or not. Every (system, test set) cell on the board carries up to six independent markers, each defined with its threshold in the methodology and recomputable from `board.json`:

| Marker | Signal | What it catches |
|---|---|---|
| **D** | Declared overlap (`data/contamination.yaml`, source quoted per entry) | the model card names the corpus as training data; the cell is excluded from the system's average by default |
| **A** | Cross-set anomaly: median-polish residual of log WER across all systems | a score far better than the system's own level and the set's difficulty predict |
| **F** | Transcript-convention fingerprint: seven orthographic rates (hamza, ta-marbuta, diacritics, punctuation, digits…) | hypotheses that mirror one corpus's annotation habits on that corpus only, the mark of training on its transcripts |
| **G** | Dev-vs-test gap: about 500 utterances from each corpus's public validation split | a system that saw the test split scores markedly better on test than on dev |
| **P** | Perturbation gap: speed ±6%, 20 dB pink noise, telephone band, on the common slice | an advantage that collapses under word-preserving perturbation, as memorized audio does |
| **L** | Likelihood probe (Whisper and Cohere decoders): NLL of the reference vs NLL of the model's own output | references the model finds more probable than its own transcription |

None of them proves contamination alone; a cell can carry several. The only signal that does not depend on what a model declares is a **private, freshly recorded set matched to each public set**; its protocol is fixed in advance in [docs/private_set_protocol.md](docs/private_set_protocol.md) and ships in v1.

## How a number is produced

1. **Inference.** Zero-shot, as released, greedy decoding, Arabic forced where the model takes a language hint. Decode settings and environment are recorded in `run.json`.
2. **Normalization.** Reference and hypothesis go through the same normalizer. `elm` is tested for string equality against the vendored original; `tolerant` adds NFC, tatweel, alef/alef-maksura/ta-marbuta/hamza folding, digit unification, and a published variant table.
3. **Scoring.** Corpus-level WER and CER per set; utterance-level percentile bootstrap (2,000 resamples) for the interval; paired bootstrap and McNemar against the baseline on shared utterances; Wilson intervals for the probes.
4. **Publication.** `scores/<model>.json` and `results/<model>/` land in the results dataset; the board renders `board.json`.

Full details: [docs/methodology.md](docs/methodology.md).

## Reproduce a row

```bash
git clone https://github.com/h9-tec/tahqiq && cd tahqiq
uv sync --extra data --extra gpu
# obtain a test set under its own terms, then
uv run python scripts/prepare_data.py casablanca
uv run python -m tahqiq.runner --model whisper_large_v3_turbo --dataset casablanca
uv run python scripts/publish.py whisper_large_v3_turbo --no-upload
```

`scripts/build.py check` refuses any run whose manifest hash differs from the committed one.

## Repository layout

```
tahqiq/            harness: normalizers, metrics, manifests, runner, model adapters, Hub I/O
scripts/           prepare_data, runner entry points, publish, timing, burst/orchestration, worker tooling
data/              manifests (with audio hashes), Elm's reference manifests, probes, model cards, inventory
space/             the static leaderboard Space
space_worker/      the ZeroGPU evaluation worker Space
docs/              methodology
tests/             normalizer parity, metrics, pipeline, adapters
```

## Scope and policy

Open-weight models only in v0. Vendors domiciled in the GCC appear only after written opt-in. Hosted APIs whose terms forbid benchmarking are excluded. All published text is numeric; the board makes no claims beyond the measured values.

## Citation

```bibtex
@misc{tahqiq2026,
  title  = {Tahqiq: a reproducible leaderboard for Arabic speech recognition},
  author = {Haroon, Hesham},
  year   = {2026},
  url    = {https://huggingface.co/spaces/HeshamHaroon/tahqiq}
}
```

## License

Code Apache-2.0. Manifests, probe files and published results CC-BY-4.0. Audio is never redistributed.
